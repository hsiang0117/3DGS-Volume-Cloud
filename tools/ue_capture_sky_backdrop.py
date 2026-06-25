"""
UE-side capture script for the viewer sky backdrop (run INSIDE Unreal's Python).

Sweeps the sun elevation 0..90 deg and, at each step, captures the visible sky
(SkyAtmosphere + ExponentialHeightFog, cloud volume hidden) into the 6 faces of
a cubemap and writes them to disk as linear-HDR images. Azimuth is NOT swept:
the sky is rotationally symmetric about the zenith except for the sun, so the
viewer rotates a single per-elevation cubemap to realise any azimuth at runtime.

Export uses 6 x SceneCapture2D (FOV 90, square RGBA16f render target) rather than
exporting a TextureRenderTargetCube directly, because RenderingLibrary.export_
render_target on an RT2D is the reliable path. The level's own SceneCaptureCube is
NOT used here (keep it for live in-editor preview, or delete it).

Capture source is SCS_SceneColorHDRNoAlpha: linear scene radiance BEFORE the
tonemapper, so auto-exposure / DoF / bloom / vignette are all bypassed. The viewer
applies the same Narkowicz tonemap it uses for the cloud, so sky and cloud share a
display space.

HOW TO RUN
  1. Open the Cloud map in the editor.
  2. Edit the CONFIG block below (OUTPUT_DIR at minimum).
  3. Window > Output Log > Cmd dropdown = "Python", then:  py "D:/3DGS-Volume-Cloud/tools/ue_capture_sky_backdrop.py"
     (or paste the file into the Python console).

OUTPUT
  OUTPUT_DIR/sky_alt{NN}_{face}{ext}   face in {px,nx,py,ny,pz,nz}, ext .exr or .hdr
  OUTPUT_DIR/sky.json                  manifest consumed by the viewer
"""
import unreal
import json
import os

# ----------------------------- CONFIG ---------------------------------------
OUTPUT_DIR   = r"D:\3DGS-Volume-Cloud\data\sky_backdrop"
FACE_SIZE    = 1024          # px per cube face (512 is plenty for a backdrop, halves disk)
ALT_MIN      = 0             # sun elevation above horizon, degrees
ALT_MAX      = 90
ALT_STEP     = 1
SUN_AZIMUTH  = None          # None -> keep the directional light's current yaw; else a float (deg)
WARMUP_CAPS  = 2             # throwaway captures per elevation so the SkyAtmosphere LUT catches up
CAPTURE_AT   = unreal.Vector(0.0, 0.0, 0.0)   # capture origin (sky is at infinity, so this is not critical)
# Faces: label -> SceneCapture2D world rotation so its forward points along the axis.
# UE is left-handed, +Z up, +X forward; +pitch tilts forward toward +Z.
FACES = [
    ("px", dict(roll=0.0, pitch=0.0,   yaw=0.0)),    # +X forward
    ("nx", dict(roll=0.0, pitch=0.0,   yaw=180.0)),  # -X
    ("py", dict(roll=0.0, pitch=0.0,   yaw=90.0)),   # +Y
    ("ny", dict(roll=0.0, pitch=0.0,   yaw=-90.0)),  # -Y
    ("pz", dict(roll=0.0, pitch=90.0,  yaw=0.0)),    # +Z up
    ("nz", dict(roll=0.0, pitch=-90.0, yaw=0.0)),    # -Z down
]
# -----------------------------------------------------------------------------


def _world():
    return unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()


def _first_actor_of(world, cls):
    found = unreal.GameplayStatics.get_all_actors_of_class(world, cls)
    return found[0] if found else None


def _all_actors_of(world, cls):
    return list(unreal.GameplayStatics.get_all_actors_of_class(world, cls))


def _try_set(comp, name, value):
    """Set an optional editor property; warn (don't abort) if the name differs by engine version."""
    try:
        comp.set_editor_property(name, value)
    except Exception as e:  # noqa: BLE001
        unreal.log_warning("optional property '{}' not set: {}".format(name, e))


def _export_face(world, rt, out_dir, base):
    """Try EXR (best), fall back to Radiance HDR. Returns the extension actually written, or None."""
    for ext in (".exr", ".hdr"):
        path = os.path.join(out_dir, base + ext)
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
        try:
            unreal.RenderingLibrary.export_render_target(world, rt, out_dir, base + ext)
        except Exception as e:  # noqa: BLE001 - engine may raise on unsupported format
            unreal.log_warning("export {}{} raised: {}".format(base, ext, e))
            continue
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return ext
    return None


def main():
    world = _world()
    if world is None:
        unreal.log_error("No editor world; open the Cloud map first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    unreal.log("UE {} | output -> {}".format(unreal.SystemLibrary.get_engine_version(), OUTPUT_DIR))

    sun = _first_actor_of(world, unreal.DirectionalLight)
    if sun is None:
        unreal.log_error("No DirectionalLight found.")
        return
    clouds = _all_actors_of(world, unreal.HeterogeneousVolume)
    unreal.log("sun={}  hidden cloud actors={}".format(sun.get_actor_label(), len(clouds)))

    orig_rot = sun.get_actor_rotation()
    azimuth = orig_rot.yaw if SUN_AZIMUTH is None else float(SUN_AZIMUTH)

    # Linear float render target so SceneColorHDR is not clipped.
    rt = unreal.RenderingLibrary.create_render_target2d(
        world, FACE_SIZE, FACE_SIZE, unreal.TextureRenderTargetFormat.RTF_RGBA16F)

    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    # Remove any temp capture actor orphaned by a previous failed run.
    for a in eas.get_all_level_actors():
        if a.get_actor_label() == "__SkyBackdropCapture_TEMP":
            eas.destroy_actor(a)

    written_ext = ".exr"
    n_ok = 0
    n_fail = 0
    cap = None
    try:
        cap = eas.spawn_actor_from_class(unreal.SceneCapture2D, CAPTURE_AT, unreal.Rotator(0, 0, 0))
        cap.set_actor_label("__SkyBackdropCapture_TEMP")
        comp = cap.capture_component2d
        comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_SCENE_COLOR_HDR_NO_ALPHA)
        comp.set_editor_property("texture_target", rt)
        comp.set_editor_property("fov_angle", 90.0)
        _try_set(comp, "capture_every_frame", False)
        _try_set(comp, "capture_on_movement", False)
        comp.set_editor_property("primitive_render_mode",
                                 unreal.SceneCapturePrimitiveRenderMode.PRM_LEGACY_SCENE_CAPTURE)
        # HiddenActors is an editor-only array that "cannot be edited on templates" for
        # a freshly spawned capture; HideActorComponents() is the runtime path that adds
        # the actor's primitives to the transient HiddenComponents list (resolved each
        # capture_scene()), which works here and gives the same result.
        if clouds:
            for c in clouds:
                try:
                    comp.hide_actor_components(c, b_skip_child_actors=False)
                except Exception as e:  # noqa: BLE001
                    unreal.log_warning("hide_actor_components failed for {}: {}".format(
                        c.get_actor_label(), e))

        alts = list(range(ALT_MIN, ALT_MAX + 1, ALT_STEP))
        for alt in alts:
            # Sun elevation -> directional light pitch = -elevation (pitch 0 = horizontal, -90 = straight down).
            sun.set_actor_rotation(unreal.Rotator(roll=0.0, pitch=-float(alt), yaw=azimuth), False)
            # Let the atmosphere LUT update for the new sun before the real captures.
            cap.set_actor_rotation(unreal.Rotator(**FACES[0][1]), False)
            for _ in range(WARMUP_CAPS):
                comp.capture_scene()
            for face, rot in FACES:
                cap.set_actor_rotation(unreal.Rotator(**rot), False)
                comp.capture_scene()
                base = "sky_alt{:02d}_{}".format(alt, face)
                ext = _export_face(world, rt, OUTPUT_DIR, base)
                if ext is None:
                    n_fail += 1
                    unreal.log_error("FAILED export {}".format(base))
                else:
                    written_ext = ext
                    n_ok += 1
            if alt % 10 == 0:
                unreal.log("  elevation {}/{} done ({} files ok)".format(alt, ALT_MAX, n_ok))

        manifest = {
            "format": "cube_faces",
            "face_size": FACE_SIZE,
            "ext": written_ext,
            "color_space": "linear_hdr",
            "capture_source": "SceneColorHDRNoAlpha",
            "alt_min": ALT_MIN, "alt_max": ALT_MAX, "alt_step": ALT_STEP,
            "file_pattern": "sky_alt{alt:02d}_{face}" + written_ext,
            "faces": [f for f, _ in FACES],
            "face_rotations": {f: r for f, r in FACES},
            "sun_azimuth_deg": azimuth,
            "ue_coordinate": "LeftHanded_Zup_Xforward",
            "fov_deg": 90.0,
        }
        with open(os.path.join(OUTPUT_DIR, "sky.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)
        unreal.log("DONE: {} files ok, {} failed. ext={}  manifest=sky.json".format(n_ok, n_fail, written_ext))
    finally:
        sun.set_actor_rotation(orig_rot, False)   # restore the sun
        if cap is not None:
            eas.destroy_actor(cap)                 # remove the temp capture actor


main()
