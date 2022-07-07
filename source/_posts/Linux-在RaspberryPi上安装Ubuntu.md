---
title: Linux | 在RaspberryPi上安装Ubuntu
date: 2022-07-04 12:46:20
tags:
  - Linux
  - Raspberry
categories:
  - linux
---

# 前言

> 参考链接：
>
> - [How to install Ubuntu Server on your Raspberry Pi | Ubuntu](https://ubuntu.com/tutorials/how-to-install-ubuntu-on-your-raspberry-pi#1-overview)
>
> 资源链接：
>
> - [SD Memory Card Formatter | SD Association (sdcard.org)](https://www.sdcard.org/downloads/formatter/)
> - [https://sourceforge.net/projects/win32diskimager/](https://sourceforge.net/projects/win32diskimager/)
> - [Install Ubuntu on a Raspberry Pi | Ubuntu](https://ubuntu.com/download/raspberry-pi)
>
> 目的：手上正好有一台树莓派，闲着也是闲着，就给它安装下系统。

# 过程

1. 系统选择

   树莓派的版本为 Raspberry Pi 4B，8G 版本的。其实装图形界面也可以，但是考虑到图形界面会消耗大量的内存，并且暂时用不上，所以系统选择为 **Ubuntu Server 22.04 LTS，64bit** 版本。

2. 格式化 SD 卡，并将系统烧录进去。

   使用 **SD Card Formatter** 将 SD 卡进行格式化。然后使用 **Win32DiskImager** 将下载的系统 img 文件烧录进 SD 卡中。

3. 配置 Wi-Fi 连接

   由于想要使用 SSH 连接系统，需要先获取 IP 地址。

   因为没有网线，无法使用以太网连接，所以我打算使用 Wi-Fi 连接。

   进入 SD 卡目录下，找到文件 `network-config`，在文件添加如下内容

   ```bash
   wifis:
     wlan0:
       dhcp4: true  # 自动分配ip
       access-points:
         "125实验室_2.4G":  # wifi名称
           password: "qwertyuiop"   # wifi密码
   ```

4. 然后启动树莓派，系统会自动连接 Wi-Fi。之后可以通过如下方式获取 IP 地址。

   - 进入路由器管理界面，查找树莓派 IP

   - 或者使用如下命令查找树莓派的 IP。

     ```bash
     # Win 系统上使用
     $ arp -a | findstr b8-27-eb dc-a6-32 e4-5f-01
     
     # Linux 系统上使用
     $ arp -na | grep -i  "b8:27:eb\|dc:a6:32\|e4:5f:01"
     ```

5. 之后便可以使用 SSH 连接树莓派。初次连接，用户名和密码都是 ubuntu。登录系统后，系统会要求更改密码。

   ```bash
   $ ssh ubuntu@<Raspberry Pi’s IP address>
   ```

# 关于静态 IP 的配置

之前的配置，使用的是动态分配 IP 的方式。导致每次 IP 地址都会发生变动。通过配置静态 IP 可以解决该问题。

系统的网络配置文件为 `/etc/netplan/50-cloud-init.yaml`，修改文件中的 Wi-Fi 配置如下。**主要是取消 DHCP，然后配置静态 IP，默认网关和 DNS 服务器。**

```bash
wifis:
  wlan0:
    # dhcp4: false   # 取消自动分配IP
    addresses:      # 静态IP
      - 192.168.1.101/24
    gateway4: 192.168.1.1   # 网关地址
    nameservers:   # DNS
      addresses: [8.8.8.8]
    optional: true
    access-points:
      "125实验室_2.4G":
        password: "qwertyuiop"
```
