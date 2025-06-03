#!/usr/bin/env python
"""
Setup S3 bucket structure for MLflow artifact storage.
"""

import boto3
import os
from botocore.exceptions import ClientError
import json


def create_bucket_structure():
    """Create the required S3 bucket structure for MLflow."""

    # Configuration
    bucket_name = os.getenv("MLFLOW_S3_BUCKET", "cashly-ai-models")
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    # Initialize S3 client
    s3_client = boto3.client('s3', region_name=region)

    print(f"🪣 Setting up S3 bucket: {bucket_name}")

    try:
        # Create bucket if it doesn't exist
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"✅ Bucket {bucket_name} already exists")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                # Create bucket
                if region == 'us-east-1':
                    s3_client.create_bucket(Bucket=bucket_name)
                else:
                    s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': region}
                    )
                print(f"✅ Created bucket {bucket_name}")

        # Create folder structure
        folders = [
            'mlflow/',                          # Root MLflow directory
            'mlflow/artifacts/',                # Model artifacts
            'mlflow/experiments/',              # Experiment data
            'mlflow/models/',                   # Registered models
            'mlflow/models/cash_flow_forecaster/',
            'mlflow/models/transaction_categorizer/',
            'mlflow/models/trend_analyzer/',
            'mlflow/models/budget_recommender/',
            'mlflow/models/anomaly_detector/',
            'data/',                           # Training data backups
            'data/raw/',
            'data/processed/',
            'checkpoints/',                    # Model checkpoints
            'logs/',                          # Training logs
        ]

        for folder in folders:
            # S3 doesn't have real folders, so we create a placeholder object
            s3_client.put_object(
                Bucket=bucket_name,
                Key=f"{folder}.keep",
                Body=b"",
                ContentType="text/plain"
            )
            print(f"📁 Created folder: {folder}")

        # Set bucket versioning
        s3_client.put_bucket_versioning(
            Bucket=bucket_name,
            VersioningConfiguration={'Status': 'Enabled'}
        )
        print("✅ Enabled bucket versioning")

        # Set lifecycle policy
        lifecycle_policy = {
            'Rules': [
                {
                    'ID': 'delete-old-artifacts',
                    'Status': 'Enabled',
                    'Prefix': 'mlflow/artifacts/',
                    'NoncurrentVersionExpiration': {
                        'NoncurrentDays': 90
                    }
                },
                {
                    'ID': 'transition-old-models',
                    'Status': 'Enabled',
                    'Prefix': 'mlflow/models/',
                    'Transitions': [
                        {
                            'Days': 30,
                            'StorageClass': 'STANDARD_IA'
                        }
                    ]
                },
                {
                    'ID': 'delete-old-logs',
                    'Status': 'Enabled',
                    'Prefix': 'logs/',
                    'Expiration': {
                        'Days': 30
                    }
                }
            ]
        }

        s3_client.put_bucket_lifecycle_configuration(
            Bucket=bucket_name,
            LifecycleConfiguration=lifecycle_policy
        )
        print("✅ Set lifecycle policies")

        # Set bucket policy for MLflow access
        bucket_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "AllowMLflowAccess",
                    "Effect": "Allow",
                    "Principal": {"AWS": f"arn:aws:iam::{boto3.client('sts').get_caller_identity()['Account']}:root"},
                    "Action": [
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:DeleteObject",
                        "s3:ListBucket"
                    ],
                    "Resource": [
                        f"arn:aws:s3:::{bucket_name}/*",
                        f"arn:aws:s3:::{bucket_name}"
                    ]
                }
            ]
        }

        s3_client.put_bucket_policy(
            Bucket=bucket_name,
            Policy=json.dumps(bucket_policy)
        )
        print("✅ Set bucket policy")

        print(f"\n🎉 S3 bucket setup complete!")
        print(f"\nBucket structure:")
        print(f"s3://{bucket_name}/")
        print("├── mlflow/")
        print("│   ├── artifacts/     # Model artifacts")
        print("│   ├── experiments/   # Experiment tracking")
        print("│   └── models/        # Model registry")
        print("├── data/              # Training data")
        print("├── checkpoints/       # Model checkpoints")
        print("└── logs/              # Training logs")

    except Exception as e:
        print(f"❌ Error setting up S3: {e}")
        raise


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv('.env.docker')

    create_bucket_structure()