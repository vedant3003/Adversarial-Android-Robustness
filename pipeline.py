import xml.etree.ElementTree as ET
import os
import subprocess
import csv

APK_DIR="APKs"
DECOMPILED_DIR="decompiled"
OUTPUT_CSV="android_features_dataset.csv"

#a small sample of permissions and intents
MASTER_PERMISSIONS = [
    "android.permission.READ_CALL_LOG",
    "android.permission.USE_CREDENTIALS",
    "android.permission.INTERNET",
    "android.permission.CAMERA",
    "android.permission.ACCESS_NETWORK_STATE"
]

MASTER_INTENTS = [
    "android.intent.action.MY_PACKAGE_REPLACED",
    "android.intent.action.MEDIA_BUTTON",
    "android.intent.action.MAIN",
    "android.intent.category.LAUNCHER",
    "android.intent.category.DEFAULT"
]

def decompile_all_apks():
    print("Starting decompilation process")
    if not os.path.exists(DECOMPILED_DIR):
        os.makedirs(DECOMPILED_DIR)

    for filename in os.listdir(APK_DIR):
        if(filename.endswith(".apk")):
            apk_path=os.path.join(APK_DIR, filename)
            out_folder_name=filename.replace(".apk","")
            out_path=os.path.join(DECOMPILED_DIR, out_folder_name)

            if(os.path.exists(out_path)):
                print(f"{filename} already decompiled")
                continue

            print(f"decompiling the apk {filename}...")
            subprocess.run(f'apktool d "{apk_path}" -o "{out_path}" -f', shell=True, input=b'\n', capture_output=True)
            print(f"Decompiled folder saved to {out_path}")

def extract_features(manifest_path):
    namespace={'android': 'http://schemas.android.com/apk/res/android'}

    try:
        tree=ET.parse(manifest_path)
        root=tree.getroot()
    except Exception as e:
        print(f"error parsing {manifest_path}:{e}")
        return None;
    present_permissions=set()
    present_intents=set()

    for perm in root.findall('.//uses-permission'):
        name = perm.get(f"{{{namespace['android']}}}name")
        if name: present_permissions.add(name)

    for action in root.findall('.//action'):
        name = action.get(f"{{{namespace['android']}}}name")
        if name: present_intents.add(name)
            
    for category in root.findall('.//category'):
        name = category.get(f"{{{namespace['android']}}}name")
        if name: present_intents.add(name)

    vector=[]
    for p in MASTER_PERMISSIONS:
        vector.append(1 if p in present_permissions else 0)
    for i in MASTER_INTENTS:
        vector.append(1 if p in present_intents else 0)

    return vector

def create_dataset():
    print("starting extaction and vectorizing of features...")
    header=["app_name"]+MASTER_PERMISSIONS+MASTER_INTENTS
    dataset_rows=[]

    for folder_name in os.listdir(DECOMPILED_DIR):
        folder_path=os.path.join(DECOMPILED_DIR,folder_name)
        manifest_path=os.path.join(folder_path, "AndroidManifest.xml")

        if os.path.exists(manifest_path):
            vector=extract_features(manifest_path)
            if vector is not None:
                row=[folder_name] + vector
                dataset_rows.append(row);
                print(f"Extacted {folder_name}")

    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer=csv.writer(file)
        writer.writerow(header)
        writer.writerows(dataset_rows)

    print(f"dataset created: {OUTPUT_CSV}")



if __name__ == "__main__":
    decompile_all_apks()
    create_dataset()