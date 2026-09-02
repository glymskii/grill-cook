"""Per-slot life sheets: what each slot looked like along its life.
Usage: python /tmp/slot_life.py smash|long"""
import sys, json, tempfile
from pathlib import Path
import cv2, numpy as np
ROOT=Path("/Users/galymzhan/Documents/Work/Vibe Coding/Grill Cook")
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"app"))
import slots
from face_reader import FaceReader
TAG=sys.argv[1]
video, dets, targets, t0, t1 = {
 "smash": ("data/IMG_6637.mov","data/mot_dets_smash_faces.json",(42,35,12,12),0,170),
 "long":  ("data/IMG_6635.mov","data/mot_dets_full_faces.json",(270,150,15,20),0,1010)}[TAG]
slots.EVENTS=Path(tempfile.mkdtemp())/"e.jsonl"
ta,tb,te,tl=targets
eng=slots.SlotEngine({"A":ta,"B":tb,"tol_early":te,"tol_late":tl}, face_model=FaceReader())
ev=[]
eng.on_event=lambda e: ev.append(dict(e, x=getattr(eng.slots.get(e['pid']),'ax',None),
                                        y=getattr(eng.slots.get(e['pid']),'ay',None),
                                        r=getattr(eng.slots.get(e['pid']),'ar',None)))
rows=json.load(open(ROOT/dets))
cap=cv2.VideoCapture(str(ROOT/video)); fps=cap.get(cv2.CAP_PROP_FPS); i=0
life={}   # sid -> dict(anchor, born, start, last, crops[(t,img)], cheesed_ts, flips[])
STEP = 8.0 if TAG=="smash" else 30.0
for r in rows:
    if r["t"]>t1: break
    want=int(r["t"]*fps)
    while i<want: cap.grab(); i+=1
    ok,f=cap.read(); i+=1
    if not ok: break
    eng.update([tuple(d) for d in r["dets"]], r["t"], f)
    fh,fw=f.shape[:2]
    for s in eng.slots.values():
        if not s.anchored: continue
        L=life.setdefault(s.sid, {"anchor":(s.ax,s.ay,s.ar),"born":s.placed_ts,"first_seen":r["t"],
                                  "crops":[],"cheesed_ts":0.0,"last":r["t"]})
        L["last"]=r["t"]; L["start"]=s.placed_ts
        if s.cheesed and not L["cheesed_ts"]: L["cheesed_ts"]=s.cheesed_ts or r["t"]
        if not L["crops"] or r["t"]-L["crops"][-1][0] >= STEP:
            R=int(s.ar*fw*1.3); cx,cy=int(s.ax*fw),int(s.ay*fh)
            c=f[max(0,cy-R):cy+R, max(0,cx-R):cx+R]
            if c.size: L["crops"].append((r["t"], cv2.resize(c,(110,110))))
cap.release()
flips={}
for e in ev:
    if e["type"]=="flip": flips.setdefault(e["pid"],[]).append(round(e["ts_video"],1))
# sheets
tiles=[]
for sid,L in sorted(life.items()):
    strip=[]
    for t,c in L["crops"][:14]:
        c=c.copy(); cv2.putText(c,f"{t:.0f}",(3,14),cv2.FONT_HERSHEY_DUPLEX,0.42,(255,255,255),1)
        if L["cheesed_ts"] and t>=L["cheesed_ts"]: cv2.rectangle(c,(0,0),(109,109),(60,200,235),2)
        for ft in flips.get(sid,[]):
            if abs(ft-t) < STEP/2: cv2.rectangle(c,(2,2),(107,107),(60,60,255),2)
        strip.append(c)
    while len(strip)<14: strip.append(np.zeros((110,110,3),np.uint8))
    row=np.hstack(strip); lab=np.zeros((22,row.shape[1],3),np.uint8)
    txt=(f"#{sid}  start {L['start']:.1f}s (born {L['first_seen']:.1f})  life {L['last']-L['start']:.0f}s  "
         f"flips {flips.get(sid,[])}  cheesed@{L['cheesed_ts']:.0f}" if L["cheesed_ts"] else
         f"#{sid}  start {L['start']:.1f}s (born {L['first_seen']:.1f})  life {L['last']-L['start']:.0f}s  flips {flips.get(sid,[])}")
    cv2.putText(lab,txt,(4,16),cv2.FONT_HERSHEY_DUPLEX,0.48,(255,255,255),1)
    tiles.append(np.vstack([lab,row]))
sheet=np.vstack(tiles) if tiles else np.zeros((10,10,3),np.uint8)
n=len(tiles); half=(n+1)//2
cv2.imwrite(f"/tmp/life_{TAG}_1.jpg", np.vstack(tiles[:half]))
if n>half: cv2.imwrite(f"/tmp/life_{TAG}_2.jpg", np.vstack(tiles[half:]))
summary={sid:{"start":round(L["start"],1),"born":round(L["first_seen"],1),"life":round(L["last"]-L["start"]),
              "anchor":[round(v,3) for v in L["anchor"]],"cheesed":round(L["cheesed_ts"]) if L["cheesed_ts"] else None,
              "flips":flips.get(sid,[])} for sid,L in life.items()}
json.dump(summary, open(f"/tmp/life_{TAG}.json","w"), indent=0)
json.dump(ev, open(str(ROOT / f"data/slot_events_{TAG}.json"),"w"))
json.dump(summary, open(str(ROOT / f"out/v9_review/life_{TAG}_new.json"),"w"), indent=0)
print(TAG, "slots:", len(life), "cheesed:", sum(1 for L in life.values() if L["cheesed_ts"]),
      "flips:", sum(len(v) for v in flips.values()))
