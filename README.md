# AI-Based Road Collision and Near-Miss Prediction Using Computer Vision

## Overview
This application analyzes road videos using a trained ResNet18 model.

## Features
- Upload road videos
- Analyze selected frames
- Display positive-class scores
- Generate annotated video
- Export CSV results

## Setup
Install dependencies using: pip install -r requirements.txt

## Run
streamlit run app.py

## Model
Place the trained checkpoint at models/improved_resnet18.pth

This is a research demonstration, not a safety-critical collision warning system.
