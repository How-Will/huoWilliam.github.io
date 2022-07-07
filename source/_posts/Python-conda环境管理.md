---
title: Python | conda环境管理
date: 2022-06-27 15:53:08
tags:
  - Conda
  - Python
categories:
  - python
---

{% timeline 2022 %}

<!-- timeline 06-29 -->

重命名，并修改bug

<!-- endtimeline -->

<!-- timeline 06-27 -->

完成写作并上传

<!-- endtimeline -->

{% endtimeline %}

# 前言

> 参考链接：
>
> - [Anaconda使用总结 - 简书 (jianshu.com)](https://www.jianshu.com/p/2f3be7781451)
>
> 目的：由于利用 conda 创建了多个虚拟环境，本文记录管理这些环境所用到的各种指令 

# 记录

1. 列出当前系统下的环境

   ```bash
   $ conda env list
   ```

2. 创建新的环境

   ```bash
   # 指定python版本为3.8，注意至少需要指定python版本或者要安装的包
   # 后一种情况下，自动安装最新python版本
   $ conda create -n env_name python=3.8
   # 同时安装必要的包
   $ conda create -n env_name numpy matplotlib python=3.8
   ```

3. 环境激活/关闭

   ```bash
   # 切换到新环境
   # linux/Mac下需要使用source activate env_name
   $ activate env_name
   #退出环境，也可以使用`activate root`切回root环境
   $ deactivate env_name
   ```

4. 删除环境

   ```bash
   $ conda remove -n env_name --all
   ```

> 用户安装的不同python环境都会被放在目录`~/anaconda/envs`下
