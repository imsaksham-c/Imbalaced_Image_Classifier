import os
import subprocess
import sys

def download_dataset():
    # Create dataset folder if it doesn't exist
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
        print("Created dataset folder")
    
    # Google Drive file ID from the provided URL
    file_id = "1ccqGu9r815WvgHAlG2CujzUPOEW_Pvo9"
    
    # Download URL
    url = f"https://drive.google.com/uc?id={file_id}"
    
    # Output filename
    output_file = "dataset/temple_dataset.zip"
    
    print("Downloading dataset...")
    
    try:
        # Use wget to download (works on most systems)
        subprocess.run([
            "wget", 
            "--no-check-certificate", 
            f"https://drive.google.com/uc?id={file_id}&export=download",
            "-O", output_file
        ], check=True)
        
        print(f"Dataset downloaded successfully to {output_file}")
        
        # Extract if it's a zip file
        if output_file.endswith('.zip'):
            print("Extracting dataset...")
            subprocess.run(["unzip", "-o", output_file, "-d", "dataset"], check=True)
            print("Dataset extracted successfully")
            
            # Remove the zip file
            os.remove(output_file)
            print("Removed zip file")
            
    except subprocess.CalledProcessError:
        print("Error downloading dataset. Please check the file ID and try again.")
        sys.exit(1)
    except FileNotFoundError:
        print("wget not found. Please install wget or use a different download method.")
        sys.exit(1)

if __name__ == "__main__":
    download_dataset() 