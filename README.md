# Personal Payment Tracker

A comprehensive solution for streamlining payment processing, receipt scanning, and payment request management.

## Project Overview

This application addresses the friction in tracking and requesting payments from friends by providing three core features:

1. **Receipt Scanning** - Automatically extract and process receipt data
2. **Payment Requests** - Send and manage payment requests to friends
3. **Bank Statement Analysis** - Monitor transactions and remind users of missed payment opportunities

## Architecture

This project follows a **domain-driven design** approach with the following structure:

```
personal-payment-tracker/
├── apps/                          # Application layer
│   ├── web/                       # Frontend web application (React/Next.js)
│   ├── mobile/                    # Mobile application (future)
│   └── api/                       # Backend API server (FastAPI)
├── packages/                      # Shared packages and libraries
│   ├── core/                      # Core business logic (Python)
│   ├── receipt_scanner/           # Receipt scanning service (Python)
│   ├── payment_requests/          # Payment request management (Python)
│   ├── bank_analyzer/             # Bank statement analysis (Python)
│   ├── shared/                    # Shared utilities and types
│   └── ui/                        # Shared UI components (React)
├── infrastructure/                # Infrastructure and deployment
│   ├── docker/                    # Docker configurations
│   ├── k8s/                       # Kubernetes manifests
│   └── terraform/                 # Infrastructure as code
├── docs/                          # Documentation
└── scripts/                       # Development and deployment scripts
```

## Technology Stack

- **Frontend**: React/Next.js with TypeScript
- **Backend**: FastAPI with Python 3.11+
- **Database**: PostgreSQL with SQLAlchemy ORM
- **File Storage**: AWS S3 or similar
- **OCR**: Google Cloud Vision API or similar
- **Authentication**: Auth0 or similar
- **Deployment**: Docker + Kubernetes

## Getting Started

See individual package READMEs for specific setup instructions.
