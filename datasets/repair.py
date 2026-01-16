import os
from pathlib import Path

# Paths
base_dir = Path("/users/student/pg/pg23/souravrout/ALL_FILES/thesis/honda/ovow/datasets")
images_dir = base_dir / "JPEGImages"
annots_dir = base_dir / "Annotations"
train_txt = base_dir / "ImageSets/Main/IDD/t1.txt"
test_txt = base_dir / "ImageSets/Main/IDD/test.txt"

def read_file_list(txt_path):
    """Read filenames from txt file"""
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def rename_files_inplace(file_list, start_idx, split_name):
    """Rename files in-place to numeric names"""
    new_names = []
    failed = []
    
    for i, old_name in enumerate(file_list):
        new_idx = start_idx + i
        new_name = f"{new_idx:06d}"
        
        old_img = images_dir / f"{old_name}.jpg"
        old_xml = annots_dir / f"{old_name}.xml"
        new_img = images_dir / f"{new_name}.jpg"
        new_xml = annots_dir / f"{new_name}.xml"
        
        try:
            # Rename image
            if old_img.exists():
                old_img.rename(new_img)
                print(f"[{split_name}] Renamed: {old_name}.jpg -> {new_name}.jpg")
            else:
                print(f"[{split_name}] SKIP: Image not found: {old_name}.jpg")
                failed.append(old_name)
                continue
            
            # Rename and update XML
            if old_xml.exists():
                # Read and update filename in XML
                with open(old_xml, 'r') as f:
                    xml_content = f.read()
                xml_content = xml_content.replace(
                    f"<filename>{old_name}.jpg</filename>",
                    f"<filename>{new_name}.jpg</filename>"
                )
                
                # Write updated XML to temp, then rename
                temp_xml = annots_dir / f"{new_name}_temp.xml"
                with open(temp_xml, 'w') as f:
                    f.write(xml_content)
                
                # Remove old, rename temp to new
                old_xml.unlink()
                temp_xml.rename(new_xml)
                print(f"[{split_name}] Renamed: {old_name}.xml -> {new_name}.xml")
            else:
                print(f"[{split_name}] SKIP: XML not found: {old_name}.xml")
                failed.append(old_name)
                continue
            
            new_names.append(new_name)
            
        except Exception as e:
            print(f"[{split_name}] ERROR processing {old_name}: {e}")
            failed.append(old_name)
    
    return new_names, failed

def main():
    print("="*70)
    print("WARNING: IN-PLACE RENAMING - NO BACKUP!")
    print("="*70)
    print("This will PERMANENTLY rename your files.")
    print("Press Ctrl+C now to cancel, or Enter to continue...")
    input()
    
    # Read splits
    print("\n[1/4] Reading split files...")
    train_files = read_file_list(train_txt)
    test_files = read_file_list(test_txt)
    print(f"Train: {len(train_files)}, Test: {len(test_files)}")
    
    # Rename train files
    print("\n[2/4] Renaming training files...")
    train_new, train_failed = rename_files_inplace(train_files, 0, "TRAIN")
    
    # Rename test files
    print("\n[3/4] Renaming test files...")
    test_new, test_failed = rename_files_inplace(test_files, len(train_files), "TEST")
    
    # Write new txt files
    print("\n[4/4] Updating split files...")
    with open(train_txt, 'w') as f:
        f.write('\n'.join(train_new))
    print(f"Updated: {train_txt}")
    
    with open(test_txt, 'w') as f:
        f.write('\n'.join(test_new))
    print(f"Updated: {test_txt}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Train: {len(train_new)} renamed, {len(train_failed)} failed")
    print(f"Test: {len(test_new)} renamed, {len(test_failed)} failed")
    
    if train_failed or test_failed:
        print("\nFailed files:")
        for f in train_failed + test_failed:
            print(f"  - {f}")
    
    print("\nDONE. Original filenames are GONE.")
    print("="*70)

if __name__ == "__main__":
    main()