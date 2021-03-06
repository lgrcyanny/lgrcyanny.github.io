title: IPsec之IP层安全架构
tags:
  - Learning
id: 312
categories:
  - Network
date: 2013-11-13 10:37:38
---

### 什么是IPSec

IP层的安全架构由IPsec定义，IPsec(Internet Protocol Security), 是一个开放标准的框架，它是为IP层提供端到端的数据加密，数据完整性和数据认证的协议簇。

IPsec最早于1995年在RFC1825和RFC1829中定义，之后在1998年IETF又重新修改为RFC2401和RFC2412, 在RFC2401中引入秘钥交换协议(IKE)管理安全联盟(Security Association, SA), 而后在2005年，IETF又发布了新的文档定义RFC4301和RFC4309，引入IKEv2管理SA。本文即基于RFC4301来阐述IPsec的主要功能和工作机制。

IPsec的设计意图是为IPv4和IPv6提供安全保护，在RFC6434之前IPsec是必选内容，而对IPv4是可选的。IPv6的地址空间广阔，更有可能成为DDos, IP欺诈等网络攻击的目标，因此IETF旨在随着IPv6的发展，推广IPsec，提高IP层的安全性。

IPsec是OSI第三层协议，它可以为上层的基于TCP和UDP协议的数据流提供数据安全保护，如SSL不能保护UDP的通信流，就可以借助IPsec保护数据流。

IPsec可以在一个主机、网关或独立的设备中进行实现，从IP层提供高质量的安全服务，包括访问控制，无连接的完整性，数据源认证，重放攻击检测和拒绝和数据加密等.

IPsec定义了IPsec作为一个IP层防火墙所需的最小功能集。IPsec进行访问控制的规则主要定义在安全规则数据库中(Security Policy Database, SPD)，IPsec根据这些规则对数据包进行保护、丢弃或通过的处理。[more...]

### IPsec安全功能组成

总体来说IPsec的安全功能由两个安全协议(AH和ESP)和建立安全联盟的协议(IKE)组成。

1\. 两种安全协议

包括认证头协议(Authentication Header，AH)和封装安全载荷协议(Encapsulating Security Payload, ESP)。

AH，提供数据完整性，反重放攻击和可选的数据源认证的安全特性。通常使用单向Hash函数的摘要算法——MD5和SHA1实现其安全特性。

ESP，可以同时提供数据完整性，数据加密，反重放攻击等安全特性。通常使用DES，3DES和AES等加密算法来进行数据加密。使用MD5和SHA1实现数据认证。

AH和ESP都采用SPD来提供访问控制。IPsec要求一定要实现ESP，AH可选，一般ESP就可以满足大部分的安全需求.

AH在实践中用的少，其原因有二：第一，没有数据加密，数据是以明文传输，而ESP提供数据加密。第二，AH提供数据源认证，一旦数据源地址发生改变，校验失败，所以AH不能穿越NAT。但是在PC到PC的通信中，采用AH更好，因为ESP的开销较大。

2\. 建立安全联盟的协议

Internet Key Exchange, IKE是用于建立和维护SA的协议，在端到端之间进行协商，确认通信端的身份，建立安全的通信关联。

在1998年RFC2401中第一次在IPsec中引入IKE，8年后RFC4301中更新为IKEv2,RFC4301中采用的是IKEv2。IKEv1和IKEv2的主要区别是：

*   IKEv2比IKEv1消耗的带宽更少。
*   IKEv2支持扩展认证头协议(Extensible Authentication Protocol, EAP), 而IKEv1不支持，IKEv2对无线的支持更好。
*   IKEv2支持MOBIKE，而IKEv1不支持，IKEv2可以提供移动通信支持。
*   IKEv2内内嵌支持NAT，而IKEv1不支持。
*   IKEv2可以检测隧道是否alive，但IKEv2不支持。
*   IKEv2的引入可以在建立安全关联时，提供更好的安全特性。

### IPsec数据处理模型

[![1.1Top Level IPsec Processing Model](http://cyanny/myblog/wp-content/uploads/2013/11/1.1Top-Level-IPsec-Processing-Model--580x344.png)](http://cyanny/myblog/wp-content/uploads/2013/11/1.1Top-Level-IPsec-Processing-Model-.png)

Psec在数据处理时,在保护接口(Protected Interface)和非保护接口(Unprotected Interface)之间创建了一个分界(Boundary)。数据从非保护接口进入，IPsec会基于AH或ESP进行访问控制，对数据的处理结果有三种：保护(Protected), 丢弃(Discard), 通过(Bypass)。

IPsec对IP数据包处理有两种情况：

Inbound，即IP输入，数据从非保护接口端进入穿过IPsec，从保护接口端输出。

Outbound, 即IP输出, 数据从保护接口端进入，从非保护接口端输出。

### IPsec两种封装模式和应用场景

IPsec的一个重要应用场景就是VPN。
[![1.2IPsec VPN应用场景](http://cyanny/myblog/wp-content/uploads/2013/11/1.2IPsec-VPN应用场景-580x321.png)](http://cyanny/myblog/wp-content/uploads/2013/11/1.2IPsec-VPN应用场景.png)

如图上图所示，企业的各个内网间通过隧道连接，而隧道模式是IPsec重要的应用模式。

IPsec提供两种封装模式，传输模式和隧道模式，主要分为四种场景：

[![1.3IPsec四种应用场景](http://cyanny/myblog/wp-content/uploads/2013/11/1.3IPsec四种应用场景-580x290.png)](http://cyanny/myblog/wp-content/uploads/2013/11/1.3IPsec四种应用场景.png)

1\. Gateway-to-Gateway网关到网关，如上图的A场景，Alice的PC要访问公司HR的服务器，需要通过隧道模式进行安全连接，该隧道就是建立在两个网关之间。这是通常的VPN的场景。

2\. End-to-Gateway端到网关，如图中的B场景，从Alice的PC端连接到另一个IPsec网关，这也是在隧道模式下。

3\. End-to-End端到端，如图中的C场景，采用隧道模式一个Cisco路由器和PIX防火墙通过隧道模式连接。

4\. 传输模式，如图中的D场景传输模式，IPsec传输模式一般用于端到端的情况，或者端到网关，此时网关被当作一个主机。D场景中Alice PC用传输模式连接PIX防火墙，对防火墙进行管理。

可以发现，IPsec的传输模式和隧道模式的区别：

传输模式一般只用于端到端的情况，如PC-to-PC。传输模式下，AH，ESP不对IP头进行修改和加密。

隧道模式可以用于任何一种场景，多用于网关到网关的场景，但是隧道模式下，AH,ESP会在源IP报文前面添加外网IP头，隧道模式会多一层IP头的开销，在端到端的模式中，建议使用传输模式。

### IPsec重要加密和认证算法

IPsec为了保证数据的完整性，实现数据的认证和加密，相关算法有：
数据加密算法：

*   DES，基于共享秘钥进行数据加密和解密，采用56bit的秘钥保证加密的性能。
*   3DES，与DES相似，提供数据加密和解密，采用变长的秘钥。
*   Diffie-Hellman(D-H): 这是一个公钥加密协议。它可以在通信的两端基于MD5或DES算法建立共享秘钥，实现安全通信。D-H主要用于IKE，建立安全session会话。秘钥一般为768bit或1024bit。
*   RSA，基于公钥的加密，来实现数据认证，例如在Cisco的路由上的IPsec采用D-H交换来决定两端的私钥，并生成共享秘钥，采用RSA签名进行公钥认证。

数据认证算法：

*   Message Digest 5（MD5）: 是一个Hash算法，用于数据包验证。这是一个单向的加密算法，IKE，AH和ESP中采用MD5进行数据认证，防止重放攻击。
*   Secure Hash Algorithm(SHA-1): 是一个Hash算法，用于数据包验证。IKE，AH和ESP中采用MD5进行数据认证，防止重放攻击。

### IPsec安全协议之AH认证头协议

AH协议主要用于保护传输分组的数据完整性，并可以进行数据源认证。它采用滑动窗口和丢弃旧数据包的技术来防止重放攻击。但是AH不提供数据加密。

在IPv4和IPv6中，AH试图保护所有的IP头字段，除了TTL，ECN, Flags等可变字段。

[![2.1AH的两种数据封装模式](http://cyanny/myblog/wp-content/uploads/2013/11/2.1AH的两种数据封装模式-580x265.png)](http://cyanny/myblog/wp-content/uploads/2013/11/2.1AH的两种数据封装模式.png)

如图，AH提供两种封装模式，传输模式和隧道模式。

传输模式下，AH采用单向Hash算法，对IP头和数据载荷进行摘要，形成AH头部，AH头部放在IP头部和数据之间。在AH处理前后IP头部不发生变化。

隧道模式下，AH对原IP头进行Hash摘要，生成新的IP头。摘要生成的头被放于新的IP头之后，原IP头之前。

AH不能通过NAT，因为NAT会改变源IP地址，破坏AH头部，造成包被IPsec另一端拒绝。

[![2.2AH头部字段](http://cyanny/myblog/wp-content/uploads/2013/11/2.2AH头部字段-580x146.png)](http://cyanny/myblog/wp-content/uploads/2013/11/2.2AH头部字段.png)

如所示是AH的头部字段，包括：

下一个头(Next Header)：8bit，标识被传输的IP报文采用的是什么协议，如TCP或UDP。

载荷长度(Payload Len)：8bit认证包头的大小。

保留字段：16bit为将来应用做保留，目前都置0。

安全参数索引(SPI): 32bit与目的IP地址一起来标识与接收方的安全关联。

序列号(Sequence Number):32bit，单调增值的序列号，防止重放攻击。

完整性检查值(Integrity Check Value, ICV): 变长字段，一般32bit的倍数，包含认证当前包所必须的数据。

### IPsec安全协议之ESP封装安全载荷协议

ESP协议为IPsec提供数据源认证，数据完整性和数据加密功能。与AH不同的是在传输模式下，ESP不提供对整个IP数据包的数据完整性和认证功能，只对IP数据载荷进行加密和认证。然而在传输模式下，提供对整个IP包的加密和认证，生成新的包头。

[![2.3ESP的两种封装模式](http://cyanny/myblog/wp-content/uploads/2013/11/2.3ESP的两种封装模式-580x229.png)](http://cyanny/myblog/wp-content/uploads/2013/11/2.3ESP的两种封装模式.png)

如图是ESP的两种封装模式。

传输模式下，ESP只对上层协议数据加密，不加密IP头，只对ESP头和加密的上层协议数据进行认证，不认证IP头，生成的新IP报文，IP头不变，ESP头和加密的数据放在IP头之后。

隧道模式下，ESP对整个IP头和上层协议数据加密，对ESP头和加密数据进行认证，生成新的IP头，包括外部IP源、目的地址。但新的IP头不参与认证。

在ESP加密和认证中，总是先对数据加密再HASH认证，这样可以让接收端在解密之前，对数据包进行快速检测，防止重放攻击。

[![2.4ESP的头部字段](http://cyanny/myblog/wp-content/uploads/2013/11/2.4ESP的头部字段-580x217.png)](http://cyanny/myblog/wp-content/uploads/2013/11/2.4ESP的头部字段.png)

如图2.4，是ESP定义的头部字段：

安全参数索引(SPI): 32bit与目的IP地址一起来标识与接收方的安全关联。

序列号(Sequence Number):32bit，单调增值的序列号，防止重放攻击。

载荷数据(Payload Data): 变长，受保护的IP数据，其数据内容由下一个头字段标识。

填充(Padding)：一些加密算法用此字段将数据填充至块的长度，和下一字段对齐。

填充长度:8bit,标识填充字段的长度。

下一个头(Next Header): 标识被传输数据所用的上层协议。

完整性检查值(Integrity Check Value, ICV): 变长字段，一般32bit的倍数，包含认证当前包所必须的数据。

### SA安全联盟

安全联盟（Security Association, SA也有翻译为安全关联、安全连接），是IPsec中的重要概念。在IPsec发送端和接收端进行安全通信之前，IPsec需要通过SA建立安全关联来确定如何建立安全通信。

IPsec为安全通信提供数据加密，数据完整性和认证这三种服务，当服务的类型确立后，通信的两端需要确立采用什么加密(DES或3DES)和认证(MD5或SHA-1)算法, 再决定算法后又需要交换session秘钥。可以对于一个安全通信会话，需要交换很多信息，所以引进SA的概念来管理安全通信参数和算法等信息。

总之，SA就是IPsec进行安全通信的一簇算法和安全参数，如封装模式，发送方和接收方的IP地址，兴趣流等，这些参数被用于加密和认证。

SA是单向的，通信的两端如果要双向通信，则需要建立一对SA，这样当任何一方的SA被破解，另一方的SA不会受影响。

[![3.1SA的结构示例](http://cyanny/myblog/wp-content/uploads/2013/11/3.1SA的结构示例-580x321.png)](http://cyanny/myblog/wp-content/uploads/2013/11/3.1SA的结构示例.png)

如图是一个SA的结构示例，包括：

目的地址

安全参数索引(SPI): 每一个通信端都有唯一的SPI，IPsec将基于SPI，源地址和目的地址在SAD数据库中搜索安全参数记录。

IPsec安全协议

秘钥

其他的SA属性，如IPsec Lifetime

[![3.2SA的工作机制](http://cyanny/myblog/wp-content/uploads/2013/11/3.2SA的工作机制-580x310.png)](http://cyanny/myblog/wp-content/uploads/2013/11/3.2SA的工作机制.png)

如图所示，是一个在具体通信中路由器R1和R2的IPsec SA参数，可以看到SA是单向的，从任何一方到另一方都要建立SA。当一个数据包需要IPsec保护，首先它会查询SAD数据库（安全关联数据库），将SA中的SPI插入到IPsec的头部，当接收端收到数据包后，接收端会根据目的地址和SPI在自己的SAD中找到相应的SA, 如果SA匹配，则进一步采用AH或ESP对数据包进行处理。

[![3.3不同通信端的不同SA](http://cyanny/myblog/wp-content/uploads/2013/11/3.3不同通信端的不同SA-580x272.png)](http://cyanny/myblog/wp-content/uploads/2013/11/3.3不同通信端的不同SA.png)

如图所示，是在一个网络中，不同的主机间的通信采用了不同IPsec安全参数，从Client到Router A采用传输模式ESP，Router A到Router B采用隧道模式AH，这些安全参数都需要SA来管理。

### SA如何判断网关的安全性----IKE

SA是安全连接，而如何建立安全连接呢？如何确定网关的安全信呢？需要因特网秘钥交换(Internet Key Exchange)协议，IKE旨在对IPsec通信的两端进行安全会话秘钥交换，建立和维护SA。

IKE的具体工作机制在RFC2409中定义，而在RFC2401中使用的IKEv2在RFC4306中有定义。

### IPsec重要数据库

1.  Security Policy Database (SPD)

    SA是管理安全通信的参数，但不管理具体的数据包的处理，对每一个IP数据包采取什么样的安全服务，如何进行访问控制，需要这些访问控制的信息就存储在安全规则数据库SPD中。

    每一个IPsec至少要实现一个SPD，SPD是一个有序数据库，它存储访问控制列表（Access Control List, ACL）,防火墙或路由的包过滤规则等。

    SPD中的记录，对数据的处理有三种选择：丢弃（Discard），通过（Bypass）和保护（Protect）。

    SPD在逻辑上被分为三部分：

    SPD-I：包括对于inbound的数据包，需要采取丢弃和通过的规则。

    SPI-S：包括需要对数据进行保护的规则。

    SPI-O：包括对于outbound的数据包，需要采取丢弃和通过的规则。

2.  Security Association Database (SAD)

    SA安全管理的相关信息，需要存储在SAD中，SAD是一个概念上的数据库，只是以某种列表的数据结构来存储SA。
    每一个实体定义了SA的安全参数。对于Outbound的数据包，SAD的实体指向SPD缓存中的SPD-S部分。对于Inbound的数据包，只通过SPI或IPsec协议来查找SA。

    具体的结构SA实体结构参见前面给出的示例.

3.  Peer Authorization Database (PAD)

    PAD，端认证数据库，提供SA管理协议，如IKE与SPD的之间的管理。有时候SA发生变化，SPD就地不到及时更新，此时就需要借助PAD对SA和SPD做出调整。

    PAD是一个有序数据库，它建立了IKE ID和SPD实体的关联。PAD支持6种ID类型：

    DNS Name

    Distinguished Name

    Email Address

    IPv4 Address

    IPv6 Address

    Key ID

    PAD的相关工作机制是：

    在IKE SA建立时，发送方和接受方都会通过IKE ID进行身份确认，确认之后向对方发送认证信息。当一个IKE信息收到后，便会通过IKE ID在PAD数据库中找到匹配的实体，每一个PAD实体都记录了进行身份认证的方法，它可以保证对不同的通信端采用正确的身份认证。

### IPsec的工作机制

[![5.1IPsec工作机制](http://cyanny/myblog/wp-content/uploads/2013/11/5.1IPsec工作机制-580x422.png)](http://cyanny/myblog/wp-content/uploads/2013/11/5.1IPsec工作机制.png)

如图5.1所示，IPsec的工作流程可以分为5个阶段：

1\. 决定兴趣流

IPsec对IP包的保护是需要消耗资源的，并非所有的流量都需要IPsec保护，那些需要IPsec保护的数据流就叫做兴趣流，IPsec会对兴趣流进行访问控制。
如发起方的兴趣流式192.168.1.0/24 -> 10.0.0.0/8，接收方的兴趣流是10.0.0.0/8->192.168.1.0/30，那么取交集，兴趣流IPsec保护的兴趣流就是192.168.1.0/30 <-> 10.0.0.0/8

IPsec通信开始时，发送方如图5.1的Router A要将兴趣流发送给接收方Router B。

2.IKE 第一步

对IPsec的通信端进行身份认证，协商并建立IKE SA。这一步建立的是IKE安全连接，通过Diffie-Hellman算法交换共享秘钥，为IKE第二步通信建立安全隧道。

3.IKE第二步

IKE第二步，主要是建立IPsec SA，在IKE第一步建立的IKE SA的基础上协商并建立IPSec SA。周期性的协商，保证SA的安全性，可选采用Diffie-Hellman交换算法。

4.IPsec数据传输

在IPsec安全隧道建立后，采用IPsec SA确立安全参数，秘钥，进行数据通信。

5.IPsec会话停止

IPsec SA通过检测和超时来停止会话，在SA停止IPsec后，如果还需要数据传输，则需要重新进入IKE第一步或IKE第二步重新协商建立新的SA。

### References

[RFC4301](http://www.rfc-editor.org/search/rfc_search_detail.php?rfc=4301&pubstatus%5B%5D=Any&pub_date_type=any)

[Cisco IPsec Overview](http://www.ciscopress.com/articles/article.asp?p=25470 "Cisco IPsec Overview")

[技术点详解---IPSec VPN基本原理](http://www.h3c.com.cn/Service/Channel_Service/Operational_Service/ICG_Technology/201005/675214_30005_0.htm)