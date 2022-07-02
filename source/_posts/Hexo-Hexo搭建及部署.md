---
title: Hexo | Hexo搭建及部署
date: 2022-06-23 19:27:34
tags: 
  - Hexo
categories: blog
---

{% timeline 2022 %}

<!-- timeline 06-29 -->

重命名

<!-- endtimeline -->

<!-- timeline 06-23 -->

完成写作并上传

<!-- endtimeline -->

{% endtimeline %}

# 前言

> 参考链接：
> 
> - [文档 | Hexo](https://hexo.io/zh-cn/docs/)
> - [GitHub+Hexo 搭建个人网站详细教程 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/26625249)
> - [【Hexo博客搭建】将其部署到GitHub Pages（二）：如何初始化并部署？-阿里云开发者社区 (aliyun.com)](https://developer.aliyun.com/article/789233?spm=a2c6h.12873639.article-detail.64.61806e0fby7lXT)
> 
> 目的：搭建个人博客，用于个人学习，生活记录。

# 过程

1. 安装Hexo
   
   > 安装前提：需要安装 Git 和 node.js
   
   使用 npm 安装 Hexo，采用局部安装的方式。创建一个文件夹，进入该文件夹后，执行如下命令
   
   ```bash
   $ npm install hexo
   ```
   
   之后便可以使用命令 `npx hexo <command>` 执行相关命令了。或者将 Hexo 所在的目录下的 `node_modules` 添加到**环境变量**之中即可直接使用 `hexo <command>`

2. 初始化
   
   执行如下命令，在指定文件建立所需要的文件
   
   ```bash
   $ hexo init <folder>
   $ cd <folder>
   $ npm install
   ```
   
   > 相关文件说明：
   > 
   > ```bash
   > .
   > ├── _config.yml   # 配置文件，进行参数配置
   > ├── package.json   # 安装包的信息。
   > ├── scaffolds  # 模板文件夹，Hexo根据该模板创建新文章
   > ├── source   # 资源文件夹是存放用户资源的地方。
   > |   ├── _drafts
   > |   └── _posts
   > └── themes  # 主题文件夹。Hexo 会根据主题来生成静态页面。
   > ```
   > 
   > 具体的配置信息：请查看 [配置 | Hexo](https://hexo.io/zh-cn/docs/configuration)

3. 查看下项目的初始状态,执行如下命令。然后浏览器输入 `localhost:4000` ，如果能看到页面，说明项目成功搭建.
   
   ```bash
   $ npm install hexo-server --save
   $ hexo server
   ```

4. 将 Hexo 项目部署到 Github 上
   
   - 创建 github 仓库。仓库名必须为 `<你的 GitHub 用户名>.github.io`
   
   - 由于 Github Pages 只支持静态文件，所以先生成静态文件。执行如下命令
     
     ```bash
     $ hexo clean && hexo generate
     ```
     
     > 站点静态文件的存储位置为 public/
   
   - 与远程仓库建立关联。先初始化 Git 仓库，然后建立一个新分支
     
     > GitHub Pages 将默认使用 main 分支作为静态文件部署。所以我们最好新建一个 hexo 分支（命名无所谓）用来存储 Hexo 的源代码，main 分支则用来存储部署后的静态文件。
     
     ```bash
     $ git init  # 在 hexo 文件夹下进行
     $ git checkout -b hexo # 建立新分支，并切换过去
     ```
   
   - 安装 [hexo-deployer-git](https://github.com/hexojs/hexo-deployer-git)，执行如下命令【**note，在站点目录下安装**】
     
     ```bash
     $ npm install hexo-deployer-git --save
     ```
   
   - 修改 `_config.yml` 文件，填入部署信息如下
     
     ```bash
     deploy:
        type: git
        repo: <repository url> #例如 <https://github.com/huoWilliam/huoWilliam.github.io.git>
        branch: [branch]  # 提交到哪个分支，默认使用main分支
        message: [message]  # 提交信息,可以自定义
     ```
   
   - 执行如下命令，完成部署
     
     ```bash
     $ hexo deploy
     ```

5. 之后便可以通过网址 `https://<你的 GitHub 用户名>.github.io` 看到你的博客。

# 备份

目前只是将生成的静态文件部署到了云端。

为了以防万一，我们应该将网站的源代码文件也推送到 GitHub 仓库备份。过程如下

1. 与远程仓库建立连接，执行如下命令
   
   ```bash
   $ git remote add origin <https://github.com/你的用户名/你的名字.github.io.git>
   ```

2. 接下来准备提交，这三句命令将是你以后每次备份所需要输入。
   
   ```bash
   $ git add -A
   $ git commit -m '描述信息'
   （第一次提交时，你可能需先运行下面命令设置一下默认提交分支）
   （git push --set-upstream origin hexo）
   $ git push
   ```
   
   > 这里同时管理了两个分支
   > 
   > - main：负责展示静态网页
   > - hexo：负责备份本地 hexo 文件
   > 
   > main 分支更新：
   > 
   > ```bash
   > $ hexo d
   > ```
   > 
   > hexo 分支更新：
   > 
   > ```bash
   > $ git add . #添加所有文件到暂存区
   > $ git commit -m "新增博客文章"  #提交
   > $ git push origin hexo #推送hexo分支到Github
   > ```
