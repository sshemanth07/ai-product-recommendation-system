#Data set is too large to be uploaded (1.3 GB in size). 
The data is therefore provided in Google drive. 
Link to the drive is:
Once downloaded, the file should be extyracted and placed in its specified format structure-Inside Data/Raw

# AI-Based Product Recommendation System

An end-to-end AI-powered product recommendation system built using the **Synerise eCommerce dataset**. This project is designed to generate personalised product recommendations based on user browsing and interaction behaviour, while also demonstrating a complete software development and MLOps workflow.

## Project Overview

This project aims to develop a recommendation system that predicts and ranks products relevant to individual users. The solution combines:

- a machine learning recommendation model,
- a **FastAPI** backend for serving predictions,
- a **Streamlit** interface for user interaction,
- **GitHub Actions** for CI/CD automation, and
- an **MLOps pipeline** for training, evaluation, versioning, and deployment.

The project also demonstrates an end-to-end SDLC process, from planning and development through testing, deployment, and maintenance.

## Objectives

The main objectives of the project are to:

- generate personalised product recommendations,
- improve recommendation quality using machine learning,
- provide a simple interface for testing recommendations,
- expose predictions through an API,
- automate testing and deployment with CI/CD, and
- support continuous model improvement through MLOps.

## Business Goals

The broader business goals defined for the project include:

- increasing advertising success and conversion performance,
- improving online commerce revenue,
- enhancing user engagement through tailored recommendations, and
- reducing customer acquisition costs through more targeted product suggestions.

## Features

### Core Features
- Personalised product recommendations
- Recommendation model trained on a public dataset
- FastAPI prediction service
- Streamlit web interface
- GitHub branching workflow
- CI/CD automation with GitHub Actions
- MLOps pipeline for retraining, evaluation, and deployment

### Nice-to-Have Features
- Analytics dashboard
- Sequential behaviour tracking
- Real-time session updates
- Cross-device mapping
- User login support for staff-facing usage

## Dataset

This project uses the **public Synerise eCommerce dataset** for model training and evaluation.

The dataset is used to:
- analyse user-item interactions,
- train recommendation models,
- evaluate recommendation quality, and
- simulate a real-world eCommerce recommendation workflow.

## Tech Stack

- **Python**
- **Pandas / NumPy** for data processing
- **Scikit-learn** and/or recommendation libraries for modelling
- **FastAPI** for model serving
- **Streamlit** for the front end
- **GitHub Actions** for CI/CD
- **ClearML** or equivalent tooling for MLOps workflow support

## System Architecture

The system follows this high-level flow:

1. **Data ingestion and preprocessing**  
   User interaction data from the Synerise dataset is cleaned and prepared.

2. **Model training and evaluation**  
   A recommendation model is trained and assessed using recommendation metrics.

3. **Prediction serving with FastAPI**  
   The trained model is exposed through an API.

4. **User interaction with Streamlit**  
   A lightweight web app allows users or reviewers to test recommendations.

5. **CI/CD and MLOps automation**  
   GitHub Actions automates testing and deployment, while the MLOps pipeline supports retraining and lifecycle management.
