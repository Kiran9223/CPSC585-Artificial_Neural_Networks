import pytesseract
import zipfile
import os
import csv
from PIL import Image
import pandas as pd

# Path to Tesseract OCR (Make sure Tesseract is installed and added to your PATH)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Modify this path as per your installation

# Function to extract text from an image
def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return ""

# Function to parse the extracted text and extract useful information
def parse_receipt_text(text):
    lines = text.splitlines()
    store_name = ""
    items = []
    total_amount = 0.0

    for line in lines:
        line = line.strip()
        if "store" in line.lower():  # You may need to adjust this based on your receipt text format
            store_name = line
        elif line:  # Assuming item details are listed here
            parts = line.split()
            if len(parts) > 1 and parts[-1].replace('.', '', 1).isdigit():  # Check if the last part is a price
                item_name = " ".join(parts[:-1])
                amount = float(parts[-1])
                items.append((item_name, amount))
                total_amount += amount

    return store_name, items, total_amount

# Function to extract all receipt files from a ZIP
def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Function to generate the shopping summary CSV file
def generate_shopping_summary_csv(store_data, output_csv):
    # Creating a dataframe to save in CSV
    all_items = []
    for store_name, items, total_amount in store_data:
        for item, amount in items:
            all_items.append([store_name, item, amount])

    df = pd.DataFrame(all_items, columns=["Store Name", "Item Name", "Amount"])
    df.to_csv(output_csv, index=False)
    print(f"Shopping summary saved to {output_csv}")

# Main Function to extract, parse, and generate the CSV
def main():
    zip_path = 'Assignment3_receipts.zip'  # Path to the zip file
    extract_to = 'receipts'  # Folder to extract images
    output_csv = 'shopping_summary.csv'  # Output CSV file

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # Extract files from ZIP
    extract_zip(zip_path, extract_to)

    store_data = []
    # Process each image in the extracted folder
    for file_name in os.listdir(extract_to):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            image_path = os.path.join(extract_to, file_name)
            print(f"Processing {image_path}...")

            # Extract text from image
            text = extract_text_from_image(image_path)
            print(text)
            # Parse the text to get store name, items, and total amount
            store_name, items, total_amount = parse_receipt_text(text)

            if store_name:
                store_data.append((store_name, items, total_amount))

    # Generate the shopping summary CSV
    generate_shopping_summary_csv(store_data, output_csv)

if __name__ == '__main__':
    main()
