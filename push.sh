#!/bin/bash
unset http_proxy && unset https_proxy
source /etc/network_turbo
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
REPO_URL="https://$TOKEN@github.com/YZC-99/Sparse-Annotations-Semantic-Segmentaion.git"
git push "$REPO_URL" main --force
echo "Script execution completed."


# git remote set-url origin https://YZC-99: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@github.com/YZC-99/Sparse-Annotations-Semantic-Segmentaion.git
