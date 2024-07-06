import os, cv2
import pandas as pd

import pytesseract
from pytesseract import Output

# pyenv virtualenv <version> <name>
# pyenv activate <name>
# pip install requirements.txt
# brew install tesseract

# Get all JPEG file paths in the 'images' folder
dir = 'img'
image_paths = [os.path.join(dir, f) for f in os.listdir(dir)]

all_data = []
for image_path in image_paths:
    print(image_path)
    
    # Load the image
    image = cv2.imread(image_path)

    # Preprocess the image: convert to grayscale and apply thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)  # Denoise while keeping edges
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Perform OCR on the image
    custom_config = r'--oem 3 --psm 6'
    details = pytesseract.image_to_data(thresh, output_type=Output.DICT, config=custom_config, lang='spa')

    # Extract and structure text into a table
    data = []
    n_boxes = len(details['level'])
    for i in range(n_boxes):
        if int(details['conf'][i]) > 20:  # Use a confidence threshold to filter out low-quality text
            (x, y, w, h) = (details['left'][i], details['top'][i], details['width'][i], details['height'][i])
            text = details['text'][i]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            data.append((x, y, w, h, text))

    # Display the image with bounding boxes
    cv2.imshow('Image with bounding boxes', image)
    cv2.waitKey(0)

    exit()

    # Sort data by y-coordinate, then x-coordinate to group text by rows
    data = sorted(data, key=lambda x: (x[1], x[0]))

    # Group text into rows based on y-coordinate
    rows = []
    current_row = []
    current_y = data[0][1]
    for item in data:
        x, y, w, h, text = item
        if y > current_y + 10:  # New row
            rows.append(current_row)
            current_row = []
            current_y = y
        current_row.append(text)
    rows.append(current_row)  # Add the last row

    # Convert rows to DataFrame
    df = pd.DataFrame(rows)

    # Append DataFrame to list
    all_data.append(df)

# Concatenate all DataFrames
final_df = pd.concat(all_data, ignore_index=True)

# Save the final DataFrame to an Excel file
excel_path = os.path.join('sheets', 'extracted_data.xlsx')
final_df.to_excel(excel_path, index=False)

print(f"Data successfully saved to {excel_path}")


# image = cv2.imread(image_paths[0])

# # Preprocess the image (optional)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# # Perform OCR on the image
# custom_config = r'--oem 3 --psm 6'
# details = pytesseract.image_to_data(gray, output_type=Output.DICT, config=custom_config)

# # Extract text and positions
# n_boxes = len(details['level'])
# for i in range(n_boxes):
#     (x, y, w, h) = (details['left'][i], details['top'][i], details['width'][i], details['height'][i])
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# # Display the image with bounding boxes
# # cv2.imshow('Image with bounding boxes', image)
# # cv2.waitKey(0)

# # Extract text and structure it as a table
# table_data = []
# for i in range(n_boxes):
#     if int(details['conf'][i]) > 60:  # Confidence level
#         text = details['text'][i]
#         (x, y, w, h) = (details['left'][i], details['top'][i], details['width'][i], details['height'][i])
#         table_data.append((x, y, w, h, text))

# # Sort and structure the extracted data
# table_data = sorted(table_data, key=lambda x: (x[1], x[0]))  # Sort by y (row), then by x (column)
# for data in table_data:
#     print(data[-1], end=' ')
#     if data[2] < 50:  # Adjust based on your table structure
#         print()