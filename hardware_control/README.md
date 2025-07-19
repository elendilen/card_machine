# 旋转拍照系统

## 功能
1. 完整360度旋转拍照（90张照片）
2. 等待接收4个角度数据
3. 根据角度精确旋转拍照（4张照片）

## 使用
1. 点击"开始旋转任务"
2. 系统完成360度旋转拍照
3. 发送角度数据到API：
   ```bash
   curl -X POST http://你的IP:5000/api/receive_angles \
     -H "Content-Type: application/json" \
     -d '{"angles": [45, 90, 135, 180]}'
   ```
4. 系统自动进行精确角度旋转拍照

## 照片输出
- `rotation_001.jpg` ~ `rotation_090.jpg` - 完整旋转照片
- `angle_1_45deg.jpg` ~ `angle_4_XXXdeg.jpg` - 精确角度照片

## 安装依赖
```bash
pip install -r requirements.txt
```
