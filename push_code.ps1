$ErrorActionPreference = "Stop"

$ignore = @"
__pycache__/
*.pyc
.pytest_cache/
tmp/
*.dll
*.so
*.log
*err*.txt
test_summary.log
"@
Set-Content -Path .gitignore -Value $ignore

git init
git config user.name "dwan-ith"
git config user.email "dw052917@gmail.com"
git branch -M main

git add .

$commitMsg = @"
Establish initial graph-native OS architecture

This commit introduces the core compiler toolchain and C runtime:
- Typed IR and ONNX digestion with explicit topological sorting
- Tensor liveness and static arena memory planner (zero-malloc)
- Zero-copy tensor aliasing and in-place destructive updates
- Generated C DAG dataflow scheduler with capability boundaries
- Heterogeneous device targeting (NPU, CPU, and DMA prefetch)
- Compile-time generated data-driven power management
"@

# Only commit if there are changes
$status = git status --porcelain
if ($status) {
    git commit -m $commitMsg
} else {
    Write-Host "No changes to commit."
}

git remote remove origin 2>$null
git remote add origin https://github.com/dwan-ith/graph-native-os.git

Write-Host "Pushing to GitHub..."
git push -u origin main
