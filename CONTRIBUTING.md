# Contributing to Digital Image Forgery Detection

Thank you for your interest in contributing! This document provides guidelines and instructions.

## 🚀 How to Contribute

### 1. Report Issues
- Check existing issues before creating a new one
- Provide clear description of the problem
- Include error messages and logs
- Specify your environment (Python version, OS, GPU info)

**Issue Template:**
```
**Description:** [Clear description of the issue]
**Steps to Reproduce:** [Steps to reproduce the behavior]
**Expected Behavior:** [What you expected to happen]
**Actual Behavior:** [What actually happened]
**Environment:** Python X.X, TensorFlow X.X.X, GPU: [Yes/No]
```

### 2. Suggest Enhancements
- Check existing discussions/issues first
- Explain the use case and benefits
- Provide examples if applicable

**Enhancement Template:**
```
**Title:** [Clear title for the feature]
**Description:** [Detailed description]
**Motivation:** [Why this feature is needed]
**Example Usage:** [Code example if possible]
```

### 3. Submit Pull Requests

#### Before You Start
1. Fork the repository
2. Create a new branch from the appropriate methodology branch:
   ```bash
   git checkout -b feature/your-feature-name
   # or for specific methodology
   git checkout experiment2
   git checkout -b feature/your-feature-name
   ```

#### Code Guidelines

**Python Style:**
- Follow PEP 8
- Use descriptive variable names
- Add docstrings to functions and classes
- Keep functions focused and modular

**Example:**
```python
def process_image(image_path: str, size: tuple = (256, 256)) -> np.ndarray:
    """
    Load and preprocess image.
    
    Args:
        image_path (str): Path to image file
        size (tuple): Target image size (height, width)
        
    Returns:
        np.ndarray: Preprocessed image array
        
    Raises:
        FileNotFoundError: If image file not found
        ValueError: If image cannot be loaded
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    img = cv2.resize(img, size)
    return img
```

**Commit Messages:**
- Use clear, descriptive messages
- Start with a verb (Add, Fix, Update, Remove, etc.)
- Keep it concise (50-72 characters)

Examples:
```
Add ELA preprocessing function
Fix ResNet50V2 model initialization bug
Update training parameters for better convergence
```

#### Submitting PR

1. **Ensure tests pass:**
   ```bash
   python -m pytest tests/
   ```

2. **Format code:**
   ```bash
   black src/
   isort src/
   flake8 src/
   ```

3. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create Pull Request** with:
   - Clear title and description
   - Reference to related issues (fixes #123)
   - Explanation of changes
   - Screenshots if applicable

**PR Template:**
```markdown
## Description
Brief explanation of changes

## Related Issues
Fixes #[issue number]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Changes Made
- Change 1
- Change 2

## How to Test
Steps to test the changes

## Screenshots (if applicable)
Before/after screenshots
```

---

## 📋 Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/Digital-image-forgery-detection.git
cd Digital-image-forgery-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install dev tools
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

---

## 🔍 Code Review Process

- Maintainers will review your PR
- You may be asked to make changes
- Once approved, your PR will be merged
- Your contribution will be credited

---

## 📊 Areas for Contribution

### High Priority
- [ ] Add unit tests for data preprocessing
- [ ] Implement model quantization for edge devices
- [ ] Create REST API for inference
- [ ] Add performance benchmarks

### Medium Priority
- [ ] Add Vision Transformer methodology
- [ ] Implement model interpretability (GradCAM)
- [ ] Add real-time video detection
- [ ] Improve documentation

### Low Priority
- [ ] Add more dataset support
- [ ] Create additional visualization tools
- [ ] Add model compression techniques

---

## 📚 Documentation

If you add new features, please document them:

1. **Code comments** for complex logic
2. **Docstrings** for functions/classes
3. **README updates** if user-facing
4. **Examples** for new functionality

---

## ❓ Questions?

- Open a discussion in the repo
- Check existing issues and PRs
- Email: abdmariam900@gmail.com

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing! 🎉
