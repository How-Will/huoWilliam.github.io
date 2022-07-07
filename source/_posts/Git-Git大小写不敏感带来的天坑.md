---
title: Git | Git大小写不敏感带来的天坑
date: 2022-07-03 18:51:19
tags:
  - Git
  - Hexo
categories:
  - git
---



{% timeline 2022 %}

<!-- timeline 07-03 -->

完成笔记，并上传

<!-- endtimeline -->

{% endtimeline %}前言

> 平台：Win11
>
> 参考链接：
>
> - [Git区分文件名大小写-阿里云开发者社区 (aliyun.com)](https://developer.aliyun.com/article/634486)
>
> 目的：Hexo 中的标签或分类如果是以大写字母开头的话，有时候访问可能出现404页面。深究原因后，发现是 Win11 上的 Git 默认对大小写不敏感，导致文件夹以大写开头的形式上传到 github 后，可能会被自动修改为小写开头。

# 过程

1. 方法1：建议文件的命名全部以小写代替，直接避免了该问题。

2. 方法2：配置 git 让它对文件名的大小写敏感

   - 配置 Git 使其对文件名大小写敏感，执行如下命令

     ```bash
     $ vim .deploy_git/.git/config  
     ```

   - 修改文件中的 `ignorecase` 字段修改为 `false`

   - 然后重新提交，执行如下命令

     ```bash
     $ git add .
     $ git commit
     $ git push origin main
     ```

     > 如果发现提交后没有生效，先执行如下命令，清理本地缓存后再重新提交。
     >
     > ```bash
     > $ git rm -r --cached .
     > ```

