# 🚀 Personal AI Agents Training System

Build your own Claude Opus-like AI agents with fine-tuning on your RTX 4050.

## 📋 Project Overview

This system trains 7 specialized AI agents over 8 weeks:

1. **Inspiration Coach** (Week 1) - Philosophical, motivational guidance
2. **Fitness Trainer** (Week 2) - Personalized workout plans
3. **Music & Poetry Generator** (Week 3) - Creative content
4. **Kaggle Data Scientist** (Week 4) - Data analysis insights
5. **Email Assistant** (Week 5) - Smart email writing
6. **Instagram Automation** (Week 6) - Social media scheduling
7. **Master Dashboard** (Week 7-8) - Coordinate all agents

---

## 🔧 Hardware Requirements

- **GPU**: NVIDIA RTX 4050 (6GB VRAM) ✅
- **RAM**: 16GB system RAM ✅
- **CPU**: Intel i7 13th Gen H ✅
- **Storage**: 1TB SSD (need ~50GB for models) ✅

---

## 📁 Project Structure

```
ai_agents_personal/
├── datasets/
│   ├── inspiration_coach_data.json
│   ├── fitness_trainer_data.json
│   └── ...more datasets
│
├── models/
│   ├── base/                           (Base model)
│   ├── mistral-7b-hf/                 (For training)
│   ├── inspiration_coach_lora/        (Trained model)
│   └── ...more trained models
│
├── agents/
│   ├── inspiration_coach.py
│   ├── test_inspiration_coach.py
│   └── ...more agents
│
├── training/
│   ├── lora_trainer.py               (Core training logic)
│   ├── train_inspiration_coach.py    (Agent-specific training)
│   └── ...more training scripts
│
├── test_setup.py                      (Verify everything works)
├── download_model.py                  (Download base model)
└── README.md                          (This file)
```

---

## ⚡ Quick Start

### Step 0: Verify Setup

```bash
python test_setup.py
```

Expected output:
```
✅ PyTorch: 2.1.2+cu121
✅ CUDA: Available
   Device: NVIDIA GeForce RTX 4050 Laptop GPU
✅ Transformers: Installed
✅ PEFT (LoRA): Installed
...
✅ ALL SYSTEMS GO!
```

### Step 1: Download Base Model

```bash
python download_model.py
```

⏱️ Takes 20-30 minutes (downloads ~9GB)

### Step 2: Train First Agent (Inspiration Coach)

```bash
cd training
python train_inspiration_coach.py
```

⏱️ Takes 4-5 hours on RTX 4050

### Step 3: Test the Model

Once training completes:

```bash
python agents/test_inspiration_coach.py
```

### Step 4: Use the Agent

```bash
python agents/inspiration_coach.py
```

You'll enter an interactive chat with your trained Inspiration Coach!

---

## 🎯 Week-by-Week Roadmap

### Week 1: Foundation & Inspiration Coach

**Days 1-2**: Setup
- [ ] Run `test_setup.py`
- [ ] Run `download_model.py`
- [ ] Verify base model downloaded (~9GB)

**Days 3-5**: Train Inspiration Coach
- [ ] Run `python training/train_inspiration_coach.py`
- [ ] Monitor training progress
- [ ] Training takes 4-5 hours

**Days 6-7**: Test & Document
- [ ] Run `python agents/test_inspiration_coach.py`
- [ ] Evaluate responses
- [ ] Use `python agents/inspiration_coach.py` interactively

### Weeks 2-7: Build More Agents

Repeat the same pattern for each agent:

1. Create dataset (`datasets/agent_name_data.json`)
2. Create training script (`training/train_agent_name.py`)
3. Create agent script (`agents/agent_name.py`)
4. Train (4-5 hours each)
5. Test and refine

### Week 8: Integration & Polish

- Connect all agents to master dashboard
- Create unified interface
- Performance optimization
- Final testing

---

## 🔄 The Training Process

### 1. Prepare Dataset

Create JSON file with training examples:

```json
[
  {
    "instruction": "Your question or task",
    "response": "The answer you want the agent to learn"
  },
  ...
]
```

Aim for:
- 500+ examples for quick training (2-3 hours)
- 1000+ examples for quality (4-5 hours)
- 1500+ examples for best results (6+ hours)

### 2. Fine-tune with LoRA

LoRA (Low-Rank Adaptation) updates only a small percentage of parameters:

```
Full Model: 7B parameters
LoRA Update: ~0.1% parameters
Training Time: 4-5 hours
VRAM Usage: ~4GB
```

### 3. Test & Iterate

```bash
python agents/test_agent_name.py
```

Quality checklist:
- ✅ Stays in character
- ✅ Provides useful responses
- ✅ No major hallucinations
- ✅ Fast (< 5 seconds per response)

---

## 📊 Training Estimates

Based on RTX 4050 performance:

| Dataset Size | Training Time | Quality |
|---|---|---|
| 100 examples | 30 minutes | Quick test |
| 500 examples | 2-3 hours | Good |
| 1000 examples | 4-5 hours | Excellent |
| 1500+ examples | 6-8 hours | Best |

**Total for 7 agents with 1000 examples each: ~35 hours**
(Spread across 8 weeks = ~4.5 hours/week)

---

## 🛠️ Troubleshooting

### Model download is slow

- Your internet connection might be slow
- Files are ~9GB total
- Can take 20-30 minutes on slow connection
- Resume from checkpoint if interrupted

### Training is running out of VRAM

RTX 4050 has 6GB VRAM. If you get CUDA errors:

1. Reduce batch size in `lora_trainer.py`:
   ```python
   train_batch_size = 1  # Already minimal
   ```

2. Reduce sequence length:
   ```python
   max_seq_length = 256  # From 512
   ```

3. Close other GPU applications

### Model generates poor responses

- Add more training data
- Train longer (increase `num_epochs` from 3 to 5)
- Improve dataset quality
- Remove low-quality examples

### CUDA out of memory during testing

GPU might be full from training. Try:
```bash
python -c "import torch; torch.cuda.empty_cache()"
```

---

## 📚 What Each File Does

### `test_setup.py`
- Verifies PyTorch installation
- Checks CUDA availability
- Tests all dependencies
- Shows training estimates

### `download_model.py`
- Downloads Mistral 7B GGUF (~4.5GB)
- Downloads HuggingFace version (~4.5GB)
- Required for training

### `training/lora_trainer.py`
- Core fine-tuning implementation
- Handles model loading, LoRA setup, training
- Provides `train()` and `generate_response()` functions

### `training/train_<agent>.py`
- Agent-specific training script
- Loads dataset, calls trainer, saves model
- Run this to train an agent

### `agents/<agent>.py`
- Use the trained agent
- Interactive chat interface
- Can be integrated with other tools

### `agents/test_<agent>.py`
- Tests model quality
- Runs sample prompts
- Shows if training was successful

---

## 🚀 Next Steps After Week 1

Once Inspiration Coach is trained:

1. **Week 2**: Create Fitness Trainer
   - Gather fitness data
   - Create `datasets/fitness_trainer_data.json`
   - Follow same training process

2. **Week 3**: Add more agents
   - Each follows same pattern
   - Should get faster at the process

3. **Week 4+**: Build commercial versions
   - Package agents
   - Add nice UI
   - Deploy for users

---

## 💡 Tips for Best Results

### Dataset Quality
- Use varied examples covering different scenarios
- Include edge cases
- Ensure responses match agent personality
- Remove duplicates and low-quality examples

### Training Parameters
- Start with default settings
- Only change if having problems
- Train for 3 epochs minimum, 5 maximum
- Higher learning rate = faster but less stable

### Agent Personality
- Define it clearly before dataset creation
- Keep it consistent throughout training examples
- Test multiple times to refine
- Iterate: retrain with improved data

---

## 📖 References

- **PyTorch**: https://pytorch.org/
- **Hugging Face**: https://huggingface.co/
- **PEFT (LoRA)**: https://github.com/huggingface/peft
- **Mistral**: https://mistral.ai/

---

## ✅ Checklist for Week 1

- [ ] PyTorch with CUDA working
- [ ] Base model downloaded
- [ ] Inspiration Coach dataset ready
- [ ] Training script running
- [ ] Model testing successful
- [ ] Interactive chat working
- [ ] Ready for Week 2

---

## 🎉 Success!

Once you complete Week 1:

```bash
python agents/inspiration_coach.py
```

You'll have a working AI agent trained on your own data!

Next: Repeat for 6 more agents over the next 7 weeks.

---

**Total investment**: ~35 hours over 8 weeks
**Result**: 7 Claude-level AI agents you fully control and can monetize

Let's go! 🚀
