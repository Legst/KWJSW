# Git 基础操作指南

Git 是一个分布式版本控制系统，广泛用于代码管理。下面介绍一些常用的 Git 基础操作，适合初学者快速上手。

---

## 1. 初始化仓库

```bash
git init
```
在当前目录创建一个新的 Git 仓库。

## 2. 配置用户信息

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```
设置提交代码时的用户名和邮箱。

## 3. 查看当前状态

```bash
git status
```
显示工作区和暂存区的状态，告诉你哪些文件被修改、添加或删除。

## 4. 添加文件到暂存区

```bash
git add 文件名
```
将文件的更改添加到暂存区准备提交。

```bash
git add .
```
添加当前目录所有变更文件到暂存区。

## 5. 提交代码

```bash
git commit -m "提交说明"
```
将暂存区的更改提交到本地仓库，并附加提交说明。

## 6. 查看提交历史

```bash
git log --oneline
```
简洁显示提交历史。

## 7. 连接远程仓库

```bash
git remote add origin 仓库地址
```
绑定本地仓库和远程仓库。

## 8. 推送代码到远程仓库

```bash
git push origin 分支名
```
将本地代码推送到远程仓库指定分支。

## 9. 拉取远程仓库代码

```bash
git pull origin 分支名
```
从远程仓库拉取最新代码并合并。

## 10. 克隆远程仓库

```bash
git clone 仓库地址
```
复制远程仓库到本地。

## 11. 创建新分支

```bash
git branch 分支名
```
## 12. 切换分支

```bash
git checkout 分支名
```
## 13. 合并分支

```bash
git merge 分支名
```
将指定分支代码合并到当前分支。

## 14. 撤销修改
撤销工作区修改（未暂存）：

```bash
git checkout -- 文件名
```
撤销暂存区的修改：

```bash
git reset HEAD 文件名
```