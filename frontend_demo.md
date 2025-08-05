# Higgs Audio v2 前端演示指南

## 快速开始

### 1. 启动测试服务器
```bash
cd /Users/luozelong/demo/higgs-audio-server
python test_frontend.py
```

### 2. 访问演示页面
浏览器会自动打开 http://localhost:3000/index.html

## 功能演示

### 📝 基础TTS演示
1. 选择"基础TTS"模式
2. 输入示例文本: "Hello, this is a demonstration of Higgs Audio v2 text-to-speech system."
3. 调整参数:
   - Temperature: 0.7
   - Top P: 0.95 
   - Top K: 50
4. 点击"生成语音"

### 🎙️ 语音克隆演示  
1. 选择"语音克隆"模式
2. 从下拉列表选择预设语音（如: belinda, en_woman等）
3. 输入克隆文本: "This is a voice cloning demonstration using Higgs Audio."
4. 可选启用"跨语言克隆"
5. 点击"克隆语音"

### 👥 多说话人对话演示
1. 选择"多说话人"模式  
2. 配置说话人语音:
   - 说话人1: en_man
   - 说话人2: en_woman
3. 输入对话文本:
```
[SPEAKER0]Hello, how are you today?
[SPEAKER1]I'm doing great, thank you for asking!
[SPEAKER0]That's wonderful to hear.
```
4. 设置分块方式为"按说话人分块"
5. 点击"生成多说话人对话"

### 🎵 实验模式演示
1. 选择"实验模式"
2. 选择模式类型:
   - 哼唱模式: 输入 "La la la, hmm hmm hmm"
   - 背景音乐: 输入 "Peaceful background music"
3. 选择参考语音
4. 点击"生成实验音频"

### 📚 长文本演示
1. 选择"长文本"模式
2. 输入长文本内容（可以是文章段落）
3. 设置分块方式: "按词数分块"
4. 设置最大词数/块: 100
5. 选择参考语音
6. 点击"生成长文本语音"

### 🎛️ 语音管理演示
1. 选择"语音管理"模式
2. 点击"刷新列表"查看系统中的语音
3. 点击"检查系统健康状态"查看服务状态
4. 如需上传新语音:
   - 输入语音名称
   - 选择音频文件
   - 点击"上传语音"

## 系统预置语音

以下是系统中可用的预置语音：

### 英文语音
- **belinda**: 女声，清晰标准
- **en_man**: 男声，沉稳
- **en_woman**: 女声，温和
- **chadwick**: 男声，深沉
- **mabel**: 女声，活泼

### 中文语音  
- **zh_man_sichuan**: 男声，四川口音
- **mabaoguo**: 特色男声

### 角色语音
- **bigbang_amy**: 《生活大爆炸》Amy角色
- **bigbang_sheldon**: 《生活大爆炸》Sheldon角色
- **shrek_shrek**: 《怪物史莱克》Shrek角色
- **shrek_fiona**: 《怪物史莱克》Fiona角色
- **shrek_donkey**: 《怪物史莱克》Donkey角色

## 测试文本示例

### 基础TTS测试文本
```
Welcome to Higgs Audio v2! This is an advanced text-to-speech system that can generate high-quality, natural-sounding speech from text input. The system supports multiple languages and various voice characteristics.
```

### 语音克隆测试文本
```
Voice cloning technology allows us to replicate the unique characteristics of a speaker's voice. This demonstration shows how effectively we can clone and generate new speech content while maintaining the original voice's distinctive features.
```

### 多说话人对话测试文本
```
[SPEAKER0]Good morning! How can I help you today?
[SPEAKER1]Hi there! I'm interested in learning about the new voice synthesis technology.
[SPEAKER0]Excellent! Our system can generate natural-sounding speech with multiple speakers.
[SPEAKER1]That sounds fascinating! Can you show me how it works?
[SPEAKER0]Of course! Let me walk you through the different features available.
```

### 长文本测试内容
```
Artificial intelligence has revolutionized many aspects of our daily lives, from the smartphones we use to the cars we drive. Voice synthesis technology, in particular, has seen tremendous advances in recent years. Modern text-to-speech systems can now generate speech that is virtually indistinguishable from human speech, opening up new possibilities for accessibility, entertainment, and communication. The Higgs Audio system represents the latest advancement in this field, offering unprecedented quality and flexibility in voice generation. With support for multiple languages, voice cloning, and multi-speaker conversations, it provides a comprehensive solution for various applications requiring high-quality synthetic speech.
```

## 故障排除

### 常见问题
1. **无法连接后端服务器**: 确保main.py服务器在localhost:8000运行
2. **语音列表为空**: 检查examples/voice_prompts目录是否存在
3. **上传文件失败**: 确认文件格式为WAV/MP3/FLAC且小于50MB
4. **生成失败**: 检查输入文本长度和参数设置

### 调试方法
1. 打开浏览器开发者工具(F12)查看控制台错误
2. 检查网络面板确认API请求状态
3. 验证服务器日志输出

## 性能提示

1. **文本长度**: 超长文本建议使用"长文本"模式
2. **参数调节**: Temperature越高随机性越强
3. **语音选择**: 选择与目标语言匹配的参考语音效果更佳
4. **网络环境**: 确保稳定的网络连接以获得最佳体验