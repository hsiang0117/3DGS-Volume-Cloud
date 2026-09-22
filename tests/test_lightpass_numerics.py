"""Small dense reference for the CUDA light-pass forward and continuous VJP."""
import argparse
import json
from pathlib import Path
import sys
import unittest

import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from diff_gaussian_rasterization import GaussianRasterizationSettings, rasterize_lightpass
from gaussian_renderer import classify_T_light
from utils.graphics_utils import getProjectionMatrix


def settings(filtered=False):
    return GaussianRasterizationSettings(image_height=10,image_width=12,tanfovx=1.,tanfovy=1.,
        bg=torch.zeros(3,device='cuda'),scale_modifier=1.,viewmatrix=torch.eye(4,device='cuda'),
        projmatrix=getProjectionMatrix(.1,20.,1.5707963267948966,1.5707963267948966).T.cuda(),
        sh_degree=0,campos=torch.zeros(3,device='cuda'),prefiltered=False,debug=True,antialiasing=False,
        light_tau_filter=filtered)


def fixture(case='normal',dtype=torch.float32):
    xyz=[[-.19,.07,2.5],[.21,-.08,3.0],[.03,.14,3.5],[-.17,.11,3.9]]
    tau=[.8,1.2,.6,.0004] if case=='normal' else [6.,3.,4.,.0004]
    scales=[[.55,.47,.37],[.62,.44,.41],[.57,.53,.32],[.61,.46,.39]]
    rot=torch.tensor([[.97,.1,.13,-.08],[.94,-.17,.08,.22],[.95,.19,-.1,.14],[.93,-.21,.16,.11]],device='cuda',dtype=dtype)
    rot=rot/torch.linalg.vector_norm(rot,dim=1,keepdim=True)
    return [torch.tensor(a,device='cuda',dtype=dtype,requires_grad=True) for a in [xyz,tau,scales]]+[rot.detach().requires_grad_(True)]


def project(inputs,sett):
    means,tau,s,q=inputs
    r,x,y,z=q.unbind(1)
    R=torch.stack([1-2*(y*y+z*z),2*(x*y-r*z),2*(x*z+r*y),
                   2*(x*y+r*z),1-2*(x*x+z*z),2*(y*z-r*x),
                   2*(x*z-r*y),2*(y*z+r*x),1-2*(x*x+y*y)],dim=1).reshape(-1,3,3)
    cov3=(R*s[:,None,:])@(R*s[:,None,:]).transpose(1,2)
    view=sett.viewmatrix.to(dtype=means.dtype)
    proj=sett.projmatrix.to(dtype=means.dtype)
    t=means@view[:3,:3]+view[3,:3]
    tx=(t[:,0]/t[:,2]).clamp(-1.3*sett.tanfovx,1.3*sett.tanfovx)*t[:,2]
    ty=(t[:,1]/t[:,2]).clamp(-1.3*sett.tanfovy,1.3*sett.tanfovy)*t[:,2]
    zero=torch.zeros_like(tx)
    fx,fy=sett.image_width/(2*sett.tanfovx),sett.image_height/(2*sett.tanfovy)
    J=torch.stack([fx/t[:,2],zero,-fx*tx/t[:,2].square(),zero,fy/t[:,2],-fy*ty/t[:,2].square()],1).reshape(-1,2,3)
    A=J@view[:3,:3].T
    cov=A@cov3@A.transpose(1,2)
    rawdet=cov[:,0,0]*cov[:,1,1]-cov[:,0,1].square()
    c=cov+torch.eye(2,device='cuda',dtype=means.dtype)*.3
    det=c[:,0,0]*c[:,1,1]-c[:,0,1].square()
    conic=torch.stack([c[:,1,1],-c[:,0,1],c[:,0,0]],dim=1)/det[:,None]
    scale=torch.sqrt(rawdet.clamp_min(0)/det) if sett.light_tau_filter else torch.ones_like(tau)
    hom=torch.cat([means,torch.ones_like(means[:,:1])],dim=1)@proj
    ndc=hom[:,:2]/(hom[:,3:4]+1e-7)
    xy=((ndc+1)*torch.tensor([sett.image_width,sett.image_height],device='cuda')-1)*.5
    return xy,conic,tau*scale,rawdet,det,scale


def reference(inputs,sett,partial=False):
    means,tau,scales,rot=inputs
    geom_inputs=[means.detach(),tau,scales.detach(),rot.detach()] if partial else inputs
    xy,conic,packed,_,_,_=project(geom_inputs,sett)
    y,x=torch.meshgrid(torch.arange(sett.image_height,device='cuda',dtype=means.dtype),
                       torch.arange(sett.image_width,device='cuda',dtype=means.dtype),indexing='ij')
    T=torch.ones_like(x)
    A=torch.zeros_like(x)
    alive=torch.ones_like(x,dtype=torch.bool)
    sums,weights,tgs,gs=[],[],[],[]
    for i in range(len(tau)):
        dx,dy=xy[i,0]-x,xy[i,1]-y
        power=-.5*(conic[i,0]*dx.square()+conic[i,2]*dy.square())-conic[i,1]*dx*dy
        G=torch.exp(power)
        reached=alive & (power<=0) & (G>0)
        tgs.append(torch.where(reached,T*G,0.).sum())
        gs.append(torch.where(reached,G,0.).sum())
        tp=packed[i]*G
        alpha=(1-torch.exp(-tp)).clamp(max=.99)
        candidate=reached & (alpha>=1/255)
        newT=T*(1-alpha)
        terminated=candidate & (newT<1e-4)
        blend=candidate & ~terminated
        w=alpha*T
        sums.append(torch.where(blend,(w.detach() if partial else w)*A,0.).sum())
        weights.append(torch.where(blend,w,0.).sum())
        A=A+torch.where(blend,tp,0.)
        T=torch.where(blend,newT,T)
        alive=alive & ~terminated
    result=[torch.stack(sums),torch.stack(weights),torch.ones_like(tau,dtype=torch.int32),torch.stack(tgs),torch.stack(gs)]
    if partial:
        for i in [1,3,4]: result[i]=result[i].detach()
    return result


def transmittance(outputs):
    s,w,r,tg,g=outputs
    return classify_T_light(w>1e-8,torch.exp(-s/w.clamp_min(1e-8)),r,tg,g)


class LightpassTests(unittest.TestCase):
    def test_forward_dense_reference_and_gradients(self):
        results=[]
        for case in ['normal','saturated']:
            for filtered in [False,True]:
                with self.subTest(case=case,filtered=filtered):
                    sett=settings(filtered)
                    cuda_in=fixture(case)
                    ref_in=[x.detach().double().requires_grad_(True) for x in cuda_in]
                    actual=rasterize_lightpass(*cuda_in,sett,full_grad=True)
                    expected=reference(ref_in,sett)
                    for idx in [0,1,3,4]:
                        torch.testing.assert_close(actual[idx].double(),expected[idx],rtol=3e-5,atol=2e-5)
                    torch.testing.assert_close(transmittance(actual).double(),transmittance(expected),rtol=3e-5,atol=2e-5)
                    # Probe both normalized T_light and each recording channel.
                    coeff=torch.tensor([.3,-.7,.4,1.1],device='cuda')
                    loss=(transmittance(actual)*coeff).sum()
                    ref_loss=(transmittance(expected)*coeff.double()).sum()
                    ga=torch.autograd.grad(loss,cuda_in,retain_graph=True)
                    gr=torch.autograd.grad(ref_loss,ref_in,retain_graph=True)
                    errs=[]
                    for a,b in zip(ga,gr):
                        errs.append(float((a.double()-b).abs().max()))
                        torch.testing.assert_close(a.double(),b,rtol=3e-3,atol=3e-5)
                    # Faint receiver fallback must receive spatial gradients.
                    self.assertGreater(float(ga[0][-1].abs().max()),1e-4)
                    # Upstream S/W/TG/G all participate; checks independent of normalization.
                    loss=sum((actual[i]*coeff*(.07+i*.03)).sum() for i in [0,1,3,4])
                    ref_loss=sum((expected[i]*coeff.double()*(.07+i*.03)).sum() for i in [0,1,3,4])
                    ga=torch.autograd.grad(loss,cuda_in)
                    gr=torch.autograd.grad(ref_loss,ref_in)
                    for a,b in zip(ga,gr):
                        torch.testing.assert_close(a.double(),b,rtol=3e-3,atol=1e-4)
                    results.append(dict(case=case,filtered=filtered,max_gradient_abs_error=errs))
        print('DENSE_REFERENCE',json.dumps(results),flush=True)

    def test_partial_control_preserves_surrogate_tau_derivative(self):
        for filtered in [False,True]:
            inputs=fixture()
            sett=settings(filtered)
            actual=rasterize_lightpass(*inputs,sett,full_grad=False)
            ref_in=[x.detach().double().requires_grad_(True) for x in inputs]
            expected=reference(ref_in,sett,partial=True)
            coeff=torch.tensor([.3,-.7,.4,1.1],device='cuda')
            a,=torch.autograd.grad((transmittance(actual)*coeff).sum(),inputs[1])
            b,=torch.autograd.grad((transmittance(expected)*coeff.double()).sum(),ref_in[1])
            torch.testing.assert_close(a.double(),b,rtol=3e-4,atol=1e-5)

    def test_filter_preserves_analytic_tau_integral(self):
        inputs=fixture()
        _,_,packed,d0,d1,h=project(inputs,settings(True))
        torch.testing.assert_close(inputs[1]*torch.sqrt(d0),packed*torch.sqrt(d1),rtol=2e-6,atol=1e-7)
        self.assertTrue(bool((h<1).all()))

    def test_finite_differences(self):
        sett=settings(True)
        inputs=fixture()
        coeff=torch.tensor([.3,-.7,.4,1.1],device='cuda')
        def loss(values):
            return (transmittance(rasterize_lightpass(*values,sett,full_grad=True))*coeff).sum()
        analytical=torch.autograd.grad(loss(inputs),inputs)
        observed=[]
        for field,index in [(0,(0,0)),(0,(2,2)),(1,(0,)),(1,(2,)),(2,(0,0)),(2,(3,1)),(3,(1,2))]:
            eps=2e-3
            plus=[x.detach().clone() for x in inputs]
            minus=[x.detach().clone() for x in inputs]
            plus[field][index]+=eps
            minus[field][index]-=eps
            numeric=float((loss(plus)-loss(minus))/(2*eps))
            auto=float(analytical[field][index])
            self.assertLess(abs(auto-numeric),2e-3*max(abs(auto),abs(numeric)) + 8e-5)
            observed.append(dict(field=field,index=index,autograd=auto,finite_difference=numeric))
        print('FINITE_DIFFERENCE',json.dumps(observed),flush=True)

    def test_empty_input(self):
        inputs=[torch.empty(shape,device='cuda',requires_grad=True) for shape in [(0,3),(0,),(0,3),(0,4)]]
        out=rasterize_lightpass(*inputs,settings(True),full_grad=True)
        grad=torch.autograd.grad(sum(out[i].sum() for i in [0,1,3,4]),inputs)
        self.assertTrue(all(g.numel()==0 for g in grad))


if __name__=='__main__':
    torch.set_num_threads(8)
    unittest.main(verbosity=2)
