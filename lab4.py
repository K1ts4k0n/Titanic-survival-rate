# titanic_survival_rate_en.py
# Titanic survival summary (English) + interactive filtering
# Shows TWO numbers now:
#   1) "Estimated survival rate" using Bayesian smoothing (robust when rows are few)
#   2) KNN (Euclidean, k=5) estimated survival rate (true distance + nearest-neighbors)
# If no rows match the filters, Bayesian falls back to a prior (group average), not 0.00%.

import pandas as pd               # ตารางข้อมูลและการประมวลผลเชิงข้อมูล
import numpy as np                # คำนวณเชิงตัวเลข/เวคเตอร์ (ใช้กับระยะทาง, KNN)
from pathlib import Path          # จัดการ path ไฟล์/โฟลเดอร์แบบ cross-platform
from sklearn.preprocessing import StandardScaler  # สเกลฟีเจอร์ให้เท่าเทียมก่อนวัดระยะทาง

# ---------- Locate train.csv ----------
def find_train_csv():
    """Find a usable path to train.csv in common locations."""
    here = Path(".")  # โฟลเดอร์ที่รันสคริปต์
    # ไล่ตรวจสามตำแหน่งยอดนิยม: ./, ./titanic_data/, ./data_titanic/
    for p in [here / "train.csv", here / "titanic_data" / "train.csv", here / "data_titanic" / "train.csv"]:
        if p.exists():       # ถ้าพบไฟล์ในตำแหน่งใด
            return str(p)    # คืน path เป็นสตริงให้ pandas ใช้อ่านได้
    # ถ้าไม่พบเลย ให้แจ้งผู้ใช้ว่าควรวางไฟล์ไว้ที่ใด
    raise FileNotFoundError(
        "train.csv not found (place it in ./ or ./titanic_data/ or ./data_titanic/)"
    )

csv_path = find_train_csv()     # ได้ path ของ train.csv
df = pd.read_csv(csv_path)      # โหลด CSV เป็น DataFrame

# ---------- Keep only the needed columns ----------
# เราใช้เฉพาะคอลัมน์ที่ต้องการ เพื่อความชัดและลดภาระหน่วยความจำ
need_cols = ["Sex", "Age", "Pclass", "Survived", "Embarked", "Cabin"]
df = df[need_cols].copy()       # เลือกคอลัมน์ + .copy() กันการเตือน SettingWithCopy

# Fill missing values — สำคัญเพื่อให้ขั้นตอนต่อไปทำงานได้ (หลีกเลี่ยง NaN)
df["Age"] = df["Age"].fillna(df["Age"].median())            # อายุ: แทนค่าหายด้วยค่ามัธยฐาน
df["Sex"] = df["Sex"].fillna(df["Sex"].mode()[0])           # เพศ: ใช้ค่าที่พบบ่อยสุด
df["Pclass"] = df["Pclass"].fillna(df["Pclass"].mode()[0])  # ชั้นโดยสาร: ใช้ค่าที่พบบ่อยสุด
df["Embarked"] = df["Embarked"].fillna("S")                 # ท่าเรือขึ้นเรือ: ปกติ default เป็น S (Southampton)
df["Cabin"] = df["Cabin"].fillna("Unknown")                 # ห้องพัก: ไม่มีให้เป็น "Unknown"

# Age bucket for readability — สร้างกลุ่มอายุให้อ่านง่ายเวลาสรุป
def age_bucket(a: float) -> str:
    """Map a numeric age to a coarse bucket label."""
    if a < 15:
        return "Child(<15)"        # เด็ก
    if a >= 60:
        return "Senior(≥60)"       # ผู้สูงอายุ
    return "Adult(15–59)"          # ผู้ใหญ่

df["AgeGroup"] = df["Age"].apply(age_bucket)  # เพิ่มคอลัมน์กลุ่มอายุ

# Deck from first cabin letter; if none, "Unknown"
def to_deck(cab: str) -> str:
    """Extract deck letter from cabin (e.g., 'C85' -> 'C'). Non-alpha/empty -> 'Unknown'."""
    s = str(cab)                                   # เผื่ออินพุตไม่ใช่สตริง
    return s[0].upper() if s and s[0].isalpha() else "Unknown"  # ตัวอักษรตัวแรกเป็น deck

df["Deck"] = df["Cabin"].apply(to_deck)           # เพิ่มคอลัมน์ Deck ที่สกัดจาก Cabin

# ---------- Quick overall summary ----------
def rate(p: float) -> str:
    """Format proportion (0..1) as percentage with 2 decimals."""
    return f"{p * 100:.2f}%"

def summarize_overall():
    """Print overall survival counts and rates."""
    total = len(df)                                 # จำนวนผู้โดยสารทั้งหมด
    survived = int(df["Survived"].sum())            # จำนวนที่รอด (รวมค่า 1 ใน Survived)
    print("\n# Overall survival")
    print(f"- Total passengers: {total}")
    print(f"- Survived: {survived}  ({rate(survived / total)})")
    print(f"- Did not survive: {total - survived}  ({rate((total - survived) / total)})")

summarize_overall()                                 # แสดงภาพรวมทันทีเมื่อรันสคริปต์

# ---------- Normalization helpers ----------
# ฟังก์ชันแปลงอินพุตผู้ใช้ให้เป็นค่ามาตรฐานสำหรับการกรอง (รองรับสะกดต่าง ๆ)
def norm_sex(s: str) -> str:
    """Normalize sex input to 'male'/'female' (accepts 'm/f' and Thai words)."""
    t = s.strip().lower()
    if t in {"male", "m"}:
        return "male"
    if t in {"female", "f"}:
        return "female"
    # accept common Thai inputs too (optional)
    if t in {"ชาย"}:
        return "male"
    if t in {"หญิง"}:
        return "female"
    return ""                                         # ค่าว่าง = ไม่ใช้ตัวกรองนี้

def norm_pclass(s: str):
    """Normalize passenger class to 1/2/3 or '' (skip)."""
    if s.strip() == "":
        return ""
    try:
        v = int(s)
        return v if v in {1, 2, 3} else ""           # อนุญาตเฉพาะ 1,2,3
    except:
        return ""                                     # แปลงไม่ได้ -> ข้าม

def norm_embarked(s: str) -> str:
    """Normalize embarkation port to C/Q/S or '' (skip). Accepts Thai names."""
    if s.strip() == "":
        return ""
    mapping_th = {"เชียร์เบิร์ก": "C", "ควีนส์ทาวน์": "Q", "เซาแธมป์ตัน": "S"}  # ชื่อไทย -> รหัส
    t = s.strip().upper()
    t = mapping_th.get(s.strip(), t)                 # ถ้าเป็นไทย แปลงเป็นรหัส; ถ้าอังกฤษอยู่แล้วก็ใช้ต่อ
    return t if t in {"C", "Q", "S"} else ""         # อนุญาตเฉพาะ C/Q/S

def norm_deck(s: str) -> str:
    """Normalize deck to A..G/T/UNKNOWN or '' (skip). Accepts 'Unknown/ไม่ทราบ'."""
    if s.strip() == "":
        return ""
    t = s.strip().upper()
    if t in {"UNKNOWN", "NOT KNOWN", "ไม่ทราบ"}:
        return "UNKNOWN"
    return t if t in {"A", "B", "C", "D", "E", "F", "G", "T", "UNKNOWN"} else ""

def ask(prompt, default=""):
    """Prompt user with a default. Empty input -> default (or 'skip')."""
    s = input(f"{prompt} (Enter = {default if default != '' else 'skip'}): ").strip()
    return s if s != "" else default

# ---------- KNN distance-based estimator (Euclidean) ----------
# ใช้ฟีเจอร์ตัวเลขง่าย ๆ 3 ตัวในการวัดระยะทาง: Age, Pclass, Sex(เข้ารหัสเป็น 0/1)
def _sex_to_num(x: str) -> int:
    """Encode sex -> numeric for distance: female=1, male/other=0."""
    return 1 if str(x).lower() == "female" else 0

# สร้าง X (ฟีเจอร์) และ y (ป้ายกำกับรอดชีวิต) จากทั้งชุด เพื่อให้มีฐานเทียบเคียงกว้าง ๆ
_knn_X = pd.DataFrame({
    "Age":    df["Age"].astype(float),
    "Pclass": df["Pclass"].astype(float),
    "SexNum": df["Sex"].apply(_sex_to_num).astype(float),
})
_knn_y = df["Survived"].astype(int).values

# สเกลมาตรฐานก่อนคำนวณ Euclidean distance (กันฟีเจอร์สเกลใหญ่ครอบงำผลระยะทาง)
_knn_scaler = StandardScaler().fit(_knn_X.values)
_knn_Xn = _knn_scaler.transform(_knn_X.values)

def knn_estimated_rate(age: float, pclass: int, sex_norm: str, k: int = 5):
    """
    ประมาณ 'อัตรารอด' ด้วย K-Nearest Neighbors:
      1) เข้ารหัส Sex -> 0/1
      2) รวมเป็นเวกเตอร์ [Age, Pclass, SexNum]
      3) สเกลด้วย StandardScaler เดียวกับที่ fit จากชุดทั้งก้อน
      4) คำนวณระยะทาง Euclidean ไปยังทุกจุด แล้วเลือก k จุดที่ใกล้สุด
      5) คืนค่า (อัตรารอดเฉลี่ยของ k เพื่อนบ้าน, ระยะทางของเพื่อนบ้าน, ดัชนีแถว)
    หมายเหตุ: Euclidean distance = sqrt(sum((x - q)^2))
    """
    q = np.array([[float(age), float(pclass), _sex_to_num(sex_norm)]], dtype=float)
    qn = _knn_scaler.transform(q)

    # ระยะทางแบบ Euclidean: sqrt(Σ (x - q)^2) ต่อแถว
    d = np.sqrt(((_knn_Xn - qn) ** 2).sum(axis=1))
    idx = np.argsort(d)[:k]           # ดึง k แถวที่ระยะสั้นสุด
    rate = _knn_y[idx].mean()         # อัตรารอด = ค่าเฉลี่ยของ Survived ในเพื่อนบ้าน k คน
    return float(rate), d[idx], idx

# ---------- Bayesian smoothing (m-estimate) ----------
# หลักการ: (k + m*prior) / (n + m)  — m ยิ่งใหญ่ ค่าจะยิ่งถูกดึงไปหา prior (ค่าเฉลี่ยกลุ่มอ้างอิง)
def bayes_smoothed_rate(k: int, n: int, prior: float, m: int = 100) -> float:
    """Return a smoothed survival rate; if n==0, return prior (not zero)."""
    if n == 0:
        return prior                                # ถ้าไม่มีข้อมูลหลังกรองเลย ให้ใช้ prior แทน
    return (k + m * prior) / (n + m)                # สูตร m-estimate

# ---------- Interactive mode (single-number output) ----------
def filter_and_show():
    """Interactive loop: read filters, compute survival-rate numbers, print them."""
    print("\n=== Compute 'Estimated survival rate' for your filters (type q to quit) ===")
    while True:  # ลูปจนกว่าจะพิมพ์ q
        go = input("\nPress Enter to input filters, or type q to quit: ").strip().lower()
        if go == "q":
            print("Done.")
            break

        # 1) รับอินพุตตัวกรองจากผู้ใช้ (สามารถกด Enter เพื่อข้ามได้)
        sex_in = ask("Sex (male/female)", "")
        pclass_in = ask("Passenger class (1/2/3)", "")
        age_in = ask("Age (e.g., 22) — optional", "")
        embarked_in = ask("Embarked (C=Cherbourg, Q=Queenstown, S=Southampton)", "")
        deck_in = ask("Deck (A/B/C/D/E/F/G/T/Unknown)", "")

        # 2) แปลงเป็นค่ามาตรฐานเพื่อใช้กรอง DataFrame
        sex = norm_sex(sex_in)
        pclass = norm_pclass(pclass_in)
        embarked = norm_embarked(embarked_in)
        deck = norm_deck(deck_in)

        # 3) เริ่มจากสำเนาข้อมูลทั้งหมด แล้วค่อย ๆ กรองตามตัวกรองที่ผู้ใช้ระบุ
        sub = df.copy()
        notes = []  # ข้อความบันทึกเหตุผล (เช่น ขยายช่วงอายุ)

        if sex:                         # ถ้ามีกรองเพศ
            sub = sub[sub["Sex"] == sex]
        if pclass != "":                # ถ้ามีกรองชั้นโดยสาร
            sub = sub[sub["Pclass"] == pclass]

        # 4) ตัวกรองอายุ: พยายามจับใกล้เคียงอายุที่ระบุ
        #    - ลองช่วง ±2 ปีก่อน
        #    - ถ้าแถวที่เข้าเงื่อนไขน้อยไป (<8) ขยายเป็น ±5 ปี
        #    - ถ้ายังน้อยมาก (<3) ใช้กลุ่มอายุ (AgeGroup) แทน
        if age_in.strip() != "":
            try:
                a = float(age_in)        # แปลงอินพุตเป็นตัวเลขอายุ
                t = 2
                sub_age = sub[(sub["Age"] >= a - t) & (sub["Age"] <= a + t)]
                if len(sub_age) < 8:     # ถ้าน้อยไป ขยายช่วง
                    t = 5
                    sub_age = sub[(sub["Age"] >= a - t) & (sub["Age"] <= a + t)]
                    notes.append(f"Used approx age window {a:.0f}±{t} years")
                sub = sub_age            # ใช้ชุดที่กรองด้วยอายุแทน
                if len(sub) < 3:         # ถ้ายังน้อยมาก ใช้กลุ่มอายุ
                    bucket = age_bucket(a)
                    sub = df.copy()      # เริ่มใหม่จากทั้งชุด เพื่อให้กลุ่มอายุมีตัวอย่างเพียงพอ
                    if sex:
                        sub = sub[sub["Sex"] == sex]
                    if pclass != "":
                        sub = sub[sub["Pclass"] == pclass]
                    if embarked:
                        sub = sub[sub["Embarked"] == embarked]
                    if deck:
                        sub = sub[sub["Deck"].str.upper() == (deck if deck != "UNKNOWN" else "UNKNOWN")]
                    sub = sub[sub["AgeGroup"] == bucket]
                    notes = [f"Used age group: {bucket}"]
            except:                       # ถ้าอายุไม่ใช่ตัวเลข
                notes.append("Invalid age: skipped age filter")

        # 5) กรองตามท่าเรือและ deck (ถ้าระบุ)
        if embarked:
            sub = sub[sub["Embarked"] == embarked]
        if deck:
            sub = sub[sub["Deck"].str.upper() == (deck if deck != "UNKNOWN" else "UNKNOWN")]

        # 6) สร้าง prior (ค่าเฉลี่ยอ้างอิง) จากกลุ่มอ้างอิง:
        #    ถ้าระบุ sex/pclass ให้คงกรองสองอันนี้ในกลุ่มอ้างอิงด้วย เพื่อได้ prior สอดคล้องบริบท
        ref = df.copy()
        if sex:
            ref = ref[ref["Sex"] == sex]
        if pclass != "":
            ref = ref[ref["Pclass"] == pclass]
        prior = ref["Survived"].mean() if len(ref) > 0 else df["Survived"].mean()

        # 7) คำนวณอัตรารอดแบบ Bayesian smoothing (หนึ่งค่าที่อ่านง่าย)
        n = len(sub)                                 # จำนวนตัวอย่างหลังกรอง
        k = int(sub["Survived"].sum()) if n > 0 else 0  # จำนวนรอดชีวิตในชุดที่กรอง
        approx = bayes_smoothed_rate(k, n, prior, m=100)  # m=100 ดึงค่าเข้า prior มากขึ้นเล็กน้อย

        # 8) สร้างข้อความสรุปตัวกรองแบบกะทัดรัด (เพื่อแสดงผล)
        conds = []
        if sex:
            conds.append(f"Sex={'female' if sex=='female' else 'male'}")
        if pclass != "":
            conds.append(f"Pclass={pclass}")
        if age_in.strip() != "":
            if notes and notes[0].startswith("Used age group"):
                conds.append(notes[0])   # ถ้าใช้กลุ่มอายุ ให้บอกชัด
            else:
                conds.append(f"Age≈{age_in}")
        if embarked:
            conds.append(f"Embarked={embarked}")
        if deck:
            conds.append(f"Deck={'Unknown' if deck=='UNKNOWN' else deck}")
        cond_txt = " | ".join(conds) if conds else "No filters (overall)"

        # 9) แสดงผลลัพธ์สุดท้าย: ตัวกรองที่ใช้ + Bayes smoothing
        print(f"\nFilters: {cond_txt}")
        print(f"Estimated survival rate (Bayesian): {approx * 100:.2f}%")

        # ----- เพิ่มการประเมินแบบ KNN (Euclidean) -----
        # เตรียมค่า age/pclass/sex สำหรับ KNN (ถ้าไม่กรอกครบ ให้ fallback เป็นค่ากลางจากกลุ่มอ้างอิง)
        try:
            age_for_knn = float(age_in) if age_in.strip() != "" else float(ref["Age"].median() if len(ref) > 0 else df["Age"].median())
        except:
            age_for_knn = float(ref["Age"].median() if len(ref) > 0 else df["Age"].median())

        if pclass != "":
            pclass_for_knn = int(pclass)
        else:
            pclass_for_knn = int(ref["Pclass"].mode().iloc[0] if len(ref) > 0 else df["Pclass"].mode().iloc[0])

        sex_for_knn = sex if sex in ("male", "female") else (ref["Sex"].mode().iloc[0] if len(ref) > 0 else df["Sex"].mode().iloc[0])

        knn_k = 5  # ปรับได้ เช่น int(len(df) ** 0.5)
        knn_rate, knn_dist, knn_idx = knn_estimated_rate(age_for_knn, pclass_for_knn, sex_for_knn, k=knn_k)

        print(f"KNN (Euclidean, k={knn_k}) estimated survival rate: {knn_rate * 100:.2f}%")
        print("Nearest distances:", np.round(knn_dist, 3))

# จุดเริ่มโปรแกรม: เข้าสู่โหมดโต้ตอบเมื่อรันไฟล์นี้โดยตรง
if __name__ == "__main__":
    filter_and_show()
