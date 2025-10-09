import cv2 as cv, numpy as np, argparse, json, os, math

def imread_f(p):
    bgr = cv.imread(p, cv.IMREAD_COLOR)
    if bgr is None: raise FileNotFoundError(p)
    return cv.cvtColor(bgr, cv.COLOR_BGR2RGB).astype(np.float32)/255.0

def imwrite_f(p, rgb):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    bgr = cv.cvtColor(np.clip(rgb,0,1), cv.COLOR_RGB2BGR)
    cv.imwrite(p, (bgr*255+0.5).astype(np.uint8))

def to_lab(rgb):
    bgr8 = (cv.cvtColor(np.clip(rgb,0,1), cv.COLOR_RGB2BGR)*255).astype(np.uint8)
    lab = cv.cvtColor(bgr8, cv.COLOR_BGR2LAB).astype(np.float32)
    return lab[:,:,0]*(100/255), lab[:,:,1]-128, lab[:,:,2]-128

def from_lab(L,a,b):
    lab = np.stack([np.clip(L*255/100,0,255), a+128, b+128], axis=-1).astype(np.uint8)
    bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    return cv.cvtColor(bgr, cv.COLOR_BGR2RGB).astype(np.float32)/255.0

def tenengrad(rgb):
    g = cv.cvtColor((rgb*255).astype(np.uint8), cv.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv.Sobel(g, cv.CV_32F,1,0,3); gy = cv.Sobel(g, cv.CV_32F,0,1,3)
    return float(np.mean(gx*gx+gy*gy))

def unsharp(rgb, amt=0.25, radius=1.0):
    blur=cv.GaussianBlur(rgb,(0,0),radius)
    return np.clip(rgb+amt*(rgb-blur),0,1)

def enhance_exposure_contrast(rgb, lift=1.2, contrast=1.2):
    L, a, b = to_lab(rgb)
    med = np.median(L)
    med = float(np.clip(med, 5.0, 95.0))  # avoid 0 or 100 blowups

    target = 70.0
    if med > 85:  # already bright enough → no lift
        gamma = 1.0
        Lg = L.copy()
    else:
        gamma = math.log(target/100 + 1e-6) / math.log(med/100 + 1e-6)
        Lg = np.power(np.clip(L/100, 0, 1), gamma) * 100

    # contrast stretch around midtones (sigmoidish)
    center = 50.0
    scale = contrast
    Lc = center + scale * (Lg - center)
    Lc = np.clip(Lc, 0, 100)

    # prevent highlights from clipping too white
    hi_clip = np.percentile(Lc, 99.5)
    if hi_clip > 95:
        Lc = Lc * (95.0 / hi_clip)

    out = from_lab(Lc, a, b)
    return out, dict(med_before=med, gamma_used=gamma)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--out_dir",required=True)
    ap.add_argument("--lift",type=float,default=1.2)
    ap.add_argument("--contrast",type=float,default=1.2)
    args=ap.parse_args()

    os.makedirs(args.out_dir,exist_ok=True)
    img=imread_f(args.input)
    imwrite_f(os.path.join(args.out_dir,"00_input.png"),img)

    out,exp=enhance_exposure_contrast(img,args.lift,args.contrast)
    sharp=tenengrad(out)
    if sharp<4000:
        out=unsharp(out,amt=0.3,radius=1.0)
    imwrite_f(os.path.join(args.out_dir,"99_final.png"),out)

    stats={"exposure":exp,"sharpness_tenengrad":sharp}
    with open(os.path.join(args.out_dir,"diagnostics.json"),"w") as f: json.dump(stats,f,indent=2)
    print(json.dumps(stats,indent=2))

if __name__=="__main__":
    main()


'''

python pro_corrector.py \
  --input /Users/karan/development/Projects/ortho/cleaned.png \
  --out_dir /Users/karan/development/Projects/ortho/corrections_boosted_v2 \
  --lift 1.2 \
  --contrast 1.3
'''