import xml.etree.ElementTree as ET
import os

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

def vectorize_manifest(manifest_path):
    print(f"Analyzing: {manifest_path}")
    
    namespace = {'android': 'http://schemas.android.com/apk/res/android'}
    
    try:
        tree = ET.parse(manifest_path)
        root = tree.getroot()
    except FileNotFoundError:
        print("Manifest file not found!")
        return None

    # 2. Extract present features from the XML
    present_permissions = set()
    present_intents = set()

    # Find all <uses-permission> tags
    for perm in root.findall('.//uses-permission'):
        name = perm.get(f"{{{namespace['android']}}}name")
        if name:
            present_permissions.add(name)

    # Find all <action> and <category> tags (these make up Intents)
    for action in root.findall('.//action'):
        name = action.get(f"{{{namespace['android']}}}name")
        if name:
            present_intents.add(name)
            
    for category in root.findall('.//category'):
        name = category.get(f"{{{namespace['android']}}}name")
        if name:
            present_intents.add(name)

    # 3. Create the Binary Vector
    feature_vector = []
    
    # Check Permissions
    for p in MASTER_PERMISSIONS:
        feature_vector.append(1 if p in present_permissions else 0)
        
    # Check Intents
    for i in MASTER_INTENTS:
        feature_vector.append(1 if i in present_intents else 0)

    return feature_vector, present_permissions, present_intents

# --- Run the Code ---
if __name__ == "__main__":
    target_manifest = "decompiled/gcalculator/AndroidManifest.xml" 
    
    vector, found_perms, found_intents = vectorize_manifest(target_manifest)
    
    if vector:
        print("\n--- Raw Features Found ---")
        print(f"Permissions: {list(found_perms)[:5]}...") 
        print(f"Intents: {list(found_intents)[:5]}...")
        
        print("\n--- Final Binary Vector ---")
        all_features = MASTER_PERMISSIONS + MASTER_INTENTS
        
        for feature, bit in zip(all_features, vector):
            print(f"[{bit}] {feature}")