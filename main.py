from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io
import uvicorn
import os
from typing import List
import json

# 鸟类类别名称（根据数据集结构）
BIRD_CLASSES = [
    "Black-footed_Albatross",
    "Brewer_Blackbird", 
    "Crested_Auklet",
    "Groove_billed_Ani",
    "Laysan_Albatross",
    "Least_Auklet",
    "Parakeet_Auklet",
    "Red_winged_Blackbird",
    "Rhinoceros_Auklet",
    "Sooty_Albatross"
]

# 模型定义（从Model.ipynb复制）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        shortcut = self.shortcut(x)
        out += shortcut
        out = F.relu(out)
        return out

class BirdResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(BirdResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.res_block1 = ResidualBlock(32, 64, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.res_block2 = ResidualBlock(128, 128, stride=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.res_block3 = ResidualBlock(256, 256, stride=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout_conv = nn.Dropout2d(p=0.3)
        self.dropout_fc = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(512 * 7 * 7, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.fc2_bn = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.res_block1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.res_block2(x)
        x = self.dropout_conv(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res_block3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 512 * 7 * 7)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout_fc(x)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# 初始化FastAPI应用
app = FastAPI(title="鸟类识别API", description="使用BirdResNet模型识别鸟类种类", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

# 图片预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model():
    """加载训练好的模型"""
    global model
    try:
        model = BirdResNet(num_classes=len(BIRD_CLASSES))
        model_path = "BirdResNet_best.pth"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")
        
        # 加载模型权重
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"✅ 模型加载成功，设备: {device}")
        return True
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    print("🚀 正在启动鸟类识别API...")
    if not load_model():
        print("⚠️ 模型加载失败，请检查模型文件是否存在")

@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "鸟类识别API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "device": str(device),
        "supported_classes": BIRD_CLASSES
    }

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.post("/predict")
async def predict_bird(file: UploadFile = File(...)):
    """
    上传图片并预测鸟类种类
    
    Args:
        file: 上传的图片文件
        
    Returns:
        JSON响应包含预测结果
    """
    if model is None:
        raise HTTPException(status_code=500, detail="模型未加载，请检查服务器状态")
    
    # 检查文件类型
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="请上传图片文件")
    
    try:
        # 读取图片
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # 转换为RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 预处理图片
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # 进行预测
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # 获取所有类别的概率
            all_probs = probabilities[0].cpu().numpy()
            
        # 构建结果
        predicted_class = BIRD_CLASSES[predicted.item()]
        confidence_score = confidence.item()
        
        # 获取前5个最可能的类别
        top5_indices = torch.topk(probabilities, 5, dim=1)[1][0].cpu().numpy()
        top5_results = []
        
        for idx in top5_indices:
            top5_results.append({
                "class": BIRD_CLASSES[idx],
                "confidence": float(all_probs[idx])
            })
        
        return JSONResponse(content={
            "success": True,
            "predicted_class": predicted_class,
            "confidence": confidence_score,
            "top5_predictions": top5_results,
            "filename": file.filename,
            "image_size": image.size
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    批量预测多张图片
    
    Args:
        files: 上传的图片文件列表
        
    Returns:
        JSON响应包含每张图片的预测结果
    """
    if model is None:
        raise HTTPException(status_code=500, detail="模型未加载，请检查服务器状态")
    
    if len(files) > 10:  # 限制批量上传数量
        raise HTTPException(status_code=400, detail="批量上传最多支持10张图片")
    
    results = []
    
    for file in files:
        try:
            # 检查文件类型
            if not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "不是有效的图片文件"
                })
                continue
            
            # 读取和预处理图片
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # 预测
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = BIRD_CLASSES[predicted.item()]
            confidence_score = confidence.item()
            
            results.append({
                "filename": file.filename,
                "success": True,
                "predicted_class": predicted_class,
                "confidence": confidence_score,
                "image_size": image.size
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return JSONResponse(content={
        "success": True,
        "total_files": len(files),
        "results": results
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
