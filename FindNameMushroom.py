api_key = ""

import google.generativeai as genai

genai.configure(api_key=api_key)
# prompt="""Tên cây nấm có các thuộc tính sau: 
#     1. cap-shape:  bell

#      2. cap-surface: smooth

#      3. cap-color:  green,

#      4. bruises:  bruises

#      5. odor:  spicy

#      6. gill-attachment: notched

#      7. gill-spacing:  close

#      8. habitat:   woods
#     Chỉ duy nhất 1 tên, không làm gì cả"""
    


def FindName(prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    print(response.text)
    
    return response.text

# print(FindName())