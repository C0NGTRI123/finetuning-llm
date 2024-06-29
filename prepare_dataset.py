import os
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def convert_8_points_to_4_points(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    x0 = min(x)
    x1 = max(x)
    y0 = min(y)
    y1 = max(y)
    line = [[x0, y0], [x1, y1]]
    return line


def sort_bbox_text_detection(bbox_lst):
    bbox_lst = sorted(bbox_lst, key=lambda x: (x[1] + x[3]))
    mean_h_bbox = np.mean([bbox[3] - bbox[1] for bbox in bbox_lst])

    line_y = bbox_lst[0][1] + bbox_lst[0][3]  # first y
    line = 1
    line_content = []

    for x1, y1, x2, y2, word, category in bbox_lst:
        if (y1 + y2) > line_y + mean_h_bbox:
            line_y = y1 + y2
            line += 1
        line_content.append([line, x1, y1, x2, y2, word, category])

    sorted_line_content = sorted(line_content)
    sorted_bbox_lst = [[x1, y1, x2, y2, word, category] for line, x1, y1, x2, y2, word, category in sorted_line_content]
    # Create text with whitespace for the same line and newline for the next line
    text = ""
    current_line = 1
    for line, x1, y1, x2, y2, word, category in sorted_line_content:
        if line == current_line:
            text += word + " "
        else:
            text += "\n" + word + " "
            current_line = line

    return sorted_bbox_lst, text.strip()


def text_from_category(bbox_lst):
    bbox_lst = sorted(bbox_lst, key=lambda x: (x[1] + x[3]))
    mean_h_bbox = np.mean([bbox[3] - bbox[1] for bbox in bbox_lst])

    line_y = bbox_lst[0][1] + bbox_lst[0][3]  # first y
    line = 1
    line_content = []

    for x1, y1, x2, y2, word, category in bbox_lst:
        if (y1 + y2) > line_y + mean_h_bbox:
            line_y = y1 + y2
            line += 1
        line_content.append([line, x1, y1, x2, y2, word, category])

    text_dict = {}
    category = sorted(list(set([cat for line, x1, y1, x2, y2, word, cat in line_content if cat != 'other'])))
    # import pdb; pdb.set_trace()
    for cat in category:
        text = ""
        for line, x1, y1, x2, y2, word, category in line_content:
            if category == cat:
                text += word + " "
        text_dict[cat] = text.strip()
    return text_dict


birthcert_folder = "/home/rb074/Downloads/20240531_gen_acte_de_naissance/llm_labelme"
birthcert_files = [f for f in sorted(os.listdir(birthcert_folder)) if f.endswith('.json')]
# import pdb; pdb.set_trace()
birthcert_json = []
for birthcert_file in birthcert_files[:]:
    birthcert_file_path = os.path.join(birthcert_folder, birthcert_file)
    with open(birthcert_file_path, "r") as f:
        birthcert_data = json.load(f)
        birthcert_data = birthcert_data['shapes']

    data_annotation = []
    for i in range(len(birthcert_data)):
        text = birthcert_data[i]['label']
        line = birthcert_data[i]['points']
        category = birthcert_data[i]['category']
        if len(line) == 4:
            line = convert_8_points_to_4_points(line)
        x0, y0 = line[0]
        x1, y1 = line[1]
        box = [x0, y0, x1, y1, text, category]
        data_annotation.append(box)

    # import pdb; pdb.set_trace()
    data_annotation, text = sort_bbox_text_detection(data_annotation)
    category_dict = text_from_category(data_annotation)
    ###### format with instruction, input, output ###########
    # instruction = "Extract the text from the document regarding the child's date of birth (only returns the original text of the date)"
    # input = f"{text}"
    # output = ""
    # for key in category_dict.keys():
    #     output += f'{key} : {category_dict[key]}\n'
    # content = {"instruction": instruction, "input": input, "output": output}
    # birthcert_json.append(content)

    ###### format with chat templates ###########
    instruction = "Extract the text from the document regarding the child's date of birth (only returns the original text of the date):"
    user = text + "\n" + instruction
    assistant = "\n".join(f"{key} : {category_dict[key]}" for key in category_dict.keys())
    user_dict = {"role": "user", "content": user}
    assistant_dict = {"role": "assistant", "content": assistant}
    conversations = [user_dict, assistant_dict]
    conversations = {"conversations": conversations}
    birthcert_json.append(conversations)

# Save the json file
with open("llm_labelme_chat.json", "w") as f:
    json.dump(birthcert_json, f, indent=4)
