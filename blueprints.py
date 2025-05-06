import streamlit as st
from openai import OpenAI
from io import BytesIO
from PIL import Image
import base64
import pandas as pd
import json
import re

def markdown_table_to_csv(markdown_text: str) -> pd.DataFrame:
    """
    Converts a valid Markdown table in a given text into a pandas DataFrame.
    It attempts to:
      1) Find continuous lines that start with '|'
      2) Parse the first row as a header
      3) Skip the second "dashes" row
      4) Parse the remaining rows as data
      5) Ensure consistent number of columns
    Raises ValueError if table is not well-formed.
    """
    lines = [line.rstrip() for line in markdown_text.split('\n')]
    table_lines = [line.strip() for line in lines if line.strip().startswith('|')]
    
    if len(table_lines) < 2:
        raise ValueError("Could not find a valid markdown table in the provided text.")
    
    def split_row(row: str, expected_cols: int):
        row = row.strip('|')
        cells = [cell.strip() for cell in row.split('|')]
        # Wypełnij brakujące komórki wartością 'N/A'
        while len(cells) < expected_cols:
            cells.append('N/A')
        # Przytnij nadmiarowe komórki
        return cells[:expected_cols]
    
    # Parsuj nagłówek i upewnij się, że ma dokładnie 3 kolumny
    expected_columns = ['Item Description', 'Quantity', 'Unit']
    header = split_row(table_lines[0], 3)
    if len(header) < 3:
        header = expected_columns[:len(header)] + ['Unit'] * (3 - len(header))
    
    data_start_idx = 1
    if len(table_lines) > 1:
        dash_chars = set(table_lines[1].replace('|','').replace(' ',''))
        if dash_chars.issubset({'-'}):
            data_start_idx = 2
    
    data_rows = []
    for row in table_lines[data_start_idx:]:
        cells = split_row(row, len(header))
        data_rows.append(cells)
    
    if not data_rows:
        raise ValueError("No valid data rows were found in the table.")
    
    return pd.DataFrame(data_rows, columns=expected_columns)

st.set_page_config(page_title='Blueprint take-off AI', page_icon='👁️')

st.markdown('# CAD Blueprint take-off AI')
api_key = st.text_input('OpenAI API Key', '', type='password')

img_input = st.file_uploader('Images', accept_multiple_files=True)

if st.button('Send'):
    if not api_key:
        st.warning('API Key required')
        st.stop()
    msg = {'role': 'user', 'content': []}
    msg['content'].append({
        'type': 'text',
        'text': (
            "Analyze the provided engineering drawing and extract a take-off of quantities for materials or components shown. "
            "Return the results ONLY as a markdown table with exactly three columns: 'Item Description', 'Quantity', and 'Unit'. "
            "Do not include any additional text, explanations, or incomplete rows. "
            "For each item, assign an appropriate unit based on the context of the drawing (e.g., 'pcs' for bolts, 'm' for beams, 'set' for assemblies, 'm²' for walls). "
            "If units are not explicitly specified in the drawing, infer them logically based on the item type. "
            "Ensure every row has a valid unit. "
            "Example:\n"
            "| Item Description        | Quantity | Unit |\n"
            "|------------------------|----------|------|\n"
            "| Concrete Beam          | 5        | m    |\n"
            "| M12 Bolts              | 10       | pcs  |\n"
            "| Steel Plate            | 2        | set  |\n"
            "| Block Wall             | 20       | m²   |"
        )
    })
    images = []
    for img in img_input:
        if img.name.split('.')[-1].lower() not in ['png', 'jpg', 'jpeg', 'gif', 'webp']:
            st.warning('Only .jpg, .png, .gif, or .webp are supported')
            st.stop()
        encoded_img = base64.b64encode(img.read()).decode('utf-8')
        images.append(img)
        msg['content'].append({
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{encoded_img}',
                'detail': 'high'
            }
        })
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model='gpt-4o',
        temperature=0.0,
        max_tokens=600,
        messages=[msg]
    )
    response_msg = str(response.choices[0].message.content)
    
    # Debug: wyświetl surową odpowiedź
    st.write("Surowa odpowiedź z API:")
    st.write(response_msg)
    
    with st.chat_message('user'):
        for i in msg['content']:
            if i['type'] == 'text':
                st.write(i['text'])
            else:
                with st.expander('Attached Image'):
                    img = Image.open(BytesIO(base64.b64decode(i['image_url']['url'][23:])))
                    st.image(img)
    
    if response_msg:
        with st.chat_message('assistant'):
            st.markdown(response_msg)
            try:
                df = markdown_table_to_csv(response_msg)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download table as CSV",
                    data=csv,
                    file_name='table.csv',
                    mime='text/csv'
                )
            except ValueError as e:
                st.error(f"Błąd podczas parsowania tabeli: {e}")
                st.markdown("Odpowiedź z API:")
                st.markdown(response_msg)
                st.stop()