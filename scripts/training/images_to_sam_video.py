import os

def fix_filenames_to_integers(img_dir, mask_dir):
    for vid_name in os.path.exists(img_dir) and os.listdir(img_dir):
        vid_img_path = os.path.join(img_dir, vid_name)
        vid_mask_path = os.path.join(mask_dir, vid_name)
        
        if not os.path.isdir(vid_img_path):
            continue
            
        # Sort files to maintain consistent frame ordering
        images = sorted(os.listdir(vid_img_path))
        
        for idx, img_file in enumerate(images):
            ext = os.path.splitext(img_file)[1] # keeps .jpg
            new_name = f"{idx:05d}{ext}"         # e.g., 00000.jpg, 00001.jpg
            
            # Rename image
            old_img = os.path.join(vid_img_path, img_file)
            new_img = os.path.join(vid_img_path, new_name)
            os.rename(old_img, new_img)
            
            # Match and rename corresponding mask
            # Assumes mask filenames match the original image filenames
            old_mask_name = img_file # or adjust if mask naming scheme differs slightly
            old_mask = os.path.join(vid_mask_path, old_mask_name)
            new_mask = os.path.join(vid_mask_path, new_name)
            
            if os.path.exists(old_mask):
                os.rename(old_mask, new_mask)

# Point to your training directories
base_path = "/home/mjhf89/ondemand/data/sys/dashboard/batch_connect/sys/jupyter/sam2/Animal_Project_Main-17/train"
fix_filenames_to_integers(os.path.join(base_path, "images"), os.path.join(base_path, "masks"))
print("File renaming complete! Frame filenames are now integers.")