import os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
ROOT = "C:/Users/t5424001/dev/tilted_optics_color_disparity_depth"
sys.path.insert(0, ROOT)
import cv2, numpy as np
import optics as op, synth as sy
from core.disparity import calcDisparityByZNCC
from core._disparity_reference import calcDisparityByZNCC as refz
SS=4
ideal_white = {"R": (1.0, 0.0), "G": (0.5, 0.5), "B": (0.0, 1.0)}
H,W_=120,320
m=np.zeros((H*SS,W_*SS),np.float32); m[:,160*SS:]=1.0
for c in [6,12,24]:
    kY,kC=sy.half_kernels(c,c,SS)
    a=sy.conv(m,kY)*2; b=sy.conv(m,kC)*2
    print("c",c,"a range",a.min(),a.max(),"sums",kY.sum(),kC.sum())
    lin={}
    for k in ("R","G","B"):
        lin[k]=sy.box_down(np.asarray(0.05+0.65*(ideal_white[k][0]*a+ideal_white[k][1]*b)),SS)
    bgr=sy.to_u8(lin)
    R=bgr[:,:,2];B=bgr[:,:,0]
    gx=cv2.convertScaleAbs(cv2.Sobel(R,cv2.CV_32F,1,0)); gb=cv2.convertScaleAbs(cv2.Sobel(B,cv2.CV_32F,1,0))
    print(" R row60",R[60,150:172]); print(" B row60",B[60,150:172])
    print(" gx row60",gx[60,150:172]); print(" gb row60",gb[60,150:172])
    d=calcDisparityByZNCC(R,B,25,41,20); dr=refz(R,B,25,41,20)
    print(" rust valid",np.isfinite(d).sum()," ref valid",np.isfinite(dr).sum(),"mask count",(gx>20).sum())
    # 1 画素で類似度列を調べる
    y,x=60,int(160+0.2*c)
    k=12;s=20
    roi=gx[y-k:y+k+1,x-k:x+k+1]
    sims=[]
    for sh in range(41):
        xx=x+s-k-sh
        rb=gb[y-k:y+k+1,xx:xx+25]
        sims.append(float(cv2.matchTemplate(roi,rb,cv2.TM_CCOEFF_NORMED)[0,0]))
    print(" sims",np.round(sims,3))
