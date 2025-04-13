import pytesseract
import zipfile
import cv2
import os
import pandas as pd
import re

# Path to Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update as per your setup

def extract_text_from_image(image_path):
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        print(text)
        return text
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return ""

def parse_receipt_text(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    store_name = ''
    items = []
    total_amount = 0.0

    # Store name detection (simplified to top lines)
    for line in lines[:5]:
        if "walmart" in line.lower() or "wal-mart" in line.lower():
            store_name = "Walmart"
            break
        elif "trader joe's" in line.lower():
            store_name = "Trader Joe's"
            break
        elif "whole foods" in line.lower():
            store_name = "Whole Foods Market"
            break
        elif "momi-toy's" in line.lower():
            store_name = "Momi Toys Creperie"
            break
        elif "costco" in line.lower():
            store_name = "Costco Wholesale"
            break
        elif "winco" in line.lower():
            store_name = "Winco"
            break
        elif "spar" in line.lower():
            store_name = "Spar"
            break

    # Patterns
    item_pattern = re.compile(r'(?:(\d+)\s+)?([a-zA-Z]+)(?:\s+(\d+))?(?:\s+([a-zA-Z]+))?\s+([\d.]+)')  
    total_pattern = re.compile(r'(TOTAL|TOTAL DUE|AMOUNT DUE|BALANCE DUE|TOTAL:)\s*\$?\s*(\d+\.\d{2})', re.IGNORECASE)

    for line in lines:
        # Check for total
        total_match = total_pattern.search(line)
        if total_match:
            try:
                total_amount = float(total_match.group(2))
            except:
                pass
        else:
            # Check for item line
            item_match = item_pattern.search(line)
            if item_match:
                item_name = item_match.group(1).strip()
                try:
                    amount = float(item_match.group(2))
                    items.append((item_name, amount))
                except:
                    continue

    return store_name, items, total_amount

def extract_zip(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    except Exception as e:
        print(f"Error extracting ZIP file: {e}")

def generate_shopping_summary_csv(store_data, output_csv):
    all_items = []

    for receipt_name, store_name, items, total_amount in store_data:
        for item, amount in items:
            all_items.append([receipt_name, store_name, item, amount])
        # Add total row for the receipt
        all_items.append([receipt_name, store_name, 'TOTAL', total_amount])

    df = pd.DataFrame(all_items, columns=["Receipt", "Store Name", "Item Name", "Amount"])
    df.to_csv(output_csv, index=False)
    print(f"\nShopping summary saved to '{output_csv}'.")

def main():
    zip_path = 'Assignment3_receipts.zip'
    extract_to = 'receipts'
    output_csv = 'shopping_summary.csv'

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    extract_zip(zip_path, extract_to)

    store_data = []

    for file_name in os.listdir(extract_to):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(extract_to, file_name)
            print(f"\n🔍 Processing {file_name}...")

            text = extract_text_from_image(image_path)
            if not text.strip():
                print(f"No text found in {file_name}")
                continue

            store_name, items, total_amount = parse_receipt_text(text)
            if store_name and items:
                store_data.append((file_name, store_name, items, total_amount))
            else:
                print(f"Could not parse relevant info from {file_name}")

    generate_shopping_summary_csv(store_data, output_csv)

if __name__ == '__main__':
    main()
