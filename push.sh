#!/bin/bash
echo "Setting up Git user..."
git config --global user.name "yzc"
git config --global user.email "yzc@qq.com"

echo "Initializing Git repository..."
git init

echo "Adding files to Git repository..."
git rm -r --cached .
git add .
git commit -m "first commit"

git add ".gitignore"
git commit -m "Update .gitignore"


echo "Setting up access token..."
TOKEN="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"

echo "Building remote repository URL..."
REPO_URL="https://$TOKEN@github.com/YZC-99/Saprse-Annotations-Semantic-Segmentaion.git"
git push "$REPO_URL" master
echo "Script execution completed."