---
title: Hexo | Hexo写作及更新
date: 2022-06-23 21:52:23
tags: 
  - Hexo
categories: blog
---

{% timeline 2022 %}

<!-- timeline 06-29 -->

重命名

<!-- endtimeline -->

<!-- timeline 06-26 -->

完成写作并上传

<!-- endtimeline -->

{% endtimeline %}

# 前言

> 参考链接：
> 
> - [【Hexo博客搭建】将其部署到GitHub Pages（三）：怎么写作以及更新？-阿里云开发者社区 (aliyun.com)](https://developer.aliyun.com/article/789409?spm=a2c6h.12873639.article-detail.63.47306e0fiO2ihC)
> 
> 目的：前面搭建完成了 Hexo，现在需要学习如何使用 Hexo 写作博客

# 过程

1. 进入博客目录下，执行如下命令，创建新文章
   
   ```bash
   $ hexo new [layout] <title>
   ```
   
   Hexo 有三种默认布局：`post`、`page`和 `draft`。在创建这三种不同类型的文件时，它们将会被保存到不同的路径；而您自定义的其他布局和 `post`相同，都将储存到 `source/_posts`文件夹。
   
   - post：存储在 source/_posts【**默认的 layout** 】
   - page：存储在source
   - draft：存储在source/_drafts

2. 进入相对应的路径下，找到新文章，使用 markdown 语法书写。

3. 更新 main 分支
   
   - 执行如下命令，生成静态文件
     
     ```bash
     $ hexo clean && hexo generate
     ```
   
   - 部署到 github 上
     
     ```bash
     $ hexo deploy # 此命令使刚刚完成写作的文章自动生成网站静态文件，并部署到设定的仓库。
     ```
     
     > 内部过程：当执行 hexo deploy 时，Hexo 会将 public 目录中的文件和目录推送至 _config.yml 中指定的远端仓库和分支中，并且完全覆盖该分支下的已有内容。

4. 对 hexo 分支更新备份，执行如下命令
   
   ```bash
   $ git add -A （此命令用来添加所有文件到暂存区）
   $ git commit -m "新增博客文章"  （此命令用来提交，双引号内可自定义内容，双引号前有空格）
   $ git push origin hexo （此命令用来推送hexo分支到Github）
   ```
