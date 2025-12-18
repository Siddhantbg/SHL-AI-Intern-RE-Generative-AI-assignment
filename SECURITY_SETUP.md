# 🔐 Security Setup Guide

## ⚠️ IMPORTANT: API Key Security

**NEVER commit API keys to git repositories!** 

## 🚀 How to Set Up Gemini API Key Securely

### For Render.com Deployment:

1. **Go to your Render.com dashboard**
2. **Select your web service**
3. **Navigate to "Environment" tab**
4. **Add environment variable:**
   - **Key:** `GEMINI_API_KEY`
   - **Value:** Your actual Gemini API key
5. **Save and redeploy**

### For Local Development:

1. **Create a `.env` file in project root:**
   ```bash
   GEMINI_API_KEY=your_actual_api_key_here
   ```

2. **Add `.env` to `.gitignore`:**
   ```bash
   echo ".env" >> .gitignore
   ```

3. **Never commit the `.env` file**

## 🔒 Security Best Practices:

- ✅ Use environment variables for all secrets
- ✅ Set API keys in deployment platform dashboards
- ✅ Add `.env` files to `.gitignore`
- ❌ Never hardcode API keys in source code
- ❌ Never commit secrets to git repositories

## 🎯 Getting Your Gemini API Key:

1. Visit: https://makersuite.google.com/app/apikey
2. Create a new API key
3. Copy the key
4. Set it as an environment variable (never commit it!)

## 🚨 If API Key is Exposed:

1. **Immediately revoke the exposed key**
2. **Generate a new API key**
3. **Update environment variables**
4. **Clean git history if needed**