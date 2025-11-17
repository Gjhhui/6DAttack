import json
import os

def fix_json_format(content):
    """修复JSON格式"""
    if not content.endswith('}'):
        content += '}'
    content = content.replace('}"annotations":', '},"annotations":')
    return content

def custom_format(data):
    """自定义格式化输出"""
    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                formatted_value = custom_format(value)
                lines.append(f'"{key}": {formatted_value}')
            else:
                lines.append(f'"{key}": {json.dumps(value)}')
        return "{\n" + ",\n".join(lines) + "\n}"
    elif isinstance(data, list):
        # 对于corner_3d和corner_2d这样的数组，保持在一行
        if data and isinstance(data[0], list):
            return json.dumps(data)
        return json.dumps(data)
    else:
        return json.dumps(data)

def find_image_info(data, image_idx):
    """查找指定索引的图片信息"""
    try:
        output_dir = "sample/output/photo_information"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        image_info = data['images'][image_idx]
        
        annotation_info = None
        for ann in data['annotations']:
            if ann['image_id'] == image_info['id']:
                annotation_info = ann
                break
                
        if image_info and annotation_info:
            result = {
                "image": {
                    "file_name": image_info['file_name'],
                    "height": image_info['height'],
                    "width": image_info['width'],
                    "id": image_info['id']
                },
                "annotation": annotation_info
            }
            
            # 只获取文件名部分，不要路径
            image_filename = os.path.basename(image_info['file_name'])  # 例如从完整路径中只取"000000.jpg"
            txt_filename = image_filename.replace('.jpg', '.txt')  # 将.jpg替换为.txt
            
            output_file = os.path.join(output_dir, txt_filename)
            
            with open(output_file, 'w') as f:
                f.write(custom_format(result))
                
            print(f"Processed and saved information for image {image_filename}")
            return result
                    
    except Exception as e:
        print(f"Error processing image {image_idx}: {e}")
        return None

def process_images_sequentially(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        content = fix_json_format(content)
        data = json.loads(content)
        
        total_images = len(data['images'])
        
        for i in range(total_images):
            find_image_info(data, i)
            
    except Exception as e:
        print(f"Error reading file: {e}")

# 执行处理
def main():
    file_path = 'sample/train.json'
    process_images_sequentially(file_path)

if __name__ == "__main__":
    main()