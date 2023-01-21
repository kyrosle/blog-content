---
title: Data-structures
tags: [algorithms]
category: Default
date: 2023-1-21
released: false
---

# 数据结构

https://www.luogu.com.cn/problem/P5490 [线段树 - 扫描线模板]
https://www.luogu.com.cn/problem/P1502 [线段树 - 扫描线变形]
https://www.luogu.com.cn/problem/P1502 [分块解决线段树问题]
https://vjudge.net/problem/HDU-1754 [分块解决线段树问题]

可持续化遇到摆烂 or 现场学~~

## tire树

### 可持续化tire树

[![trie.001.jpeg](https://kyros.oss-cn-shenzhen.aliyuncs.com/markdown/8330_69574ea4a4-trie.001.jpeg?x-oss-process=style/kyros)](https://kyros.oss-cn-shenzhen.aliyuncs.com/markdown/8330_69574ea4a4-trie.001.jpeg?x-oss-process=style/kyros)

```
C++
#include <algorithm>
#include <ios>
#include <iostream>
using namespace std;
// 定义大小
const int N = 6e5 + 10;
const int M = N * 25;
int tr[M][2]; // 存指针 
int root[N], idx;// 
int s[N];
int max_id[M];
int n, m;
-----
    i -> 第几个版本的tire树
    k -> 当前的操作数 23 - 0 ==> 1e7 ~ 23
    p -> 前一版本的tire树
    q -> 新构造的tire树
void insert(int i, int k, int p, int q)
{
    // 到头了
    if(k < 0)
    {
        max_id[q] = i; 
        return;
    }
    // 取出位置上0/1
    int v = s[i] >> k & 1;
    // pre_tire 拼接 到 新 tire树
    if(p)
        tr[q][v ^ 1] = tr[p][v ^ 1];
    // 开点
    tr[q][v] = ++idx;
    // 递归处理
    insert(i, k - 1, tr[p][v], tr[q][v]);

    max_id[q] = max(max_id[tr[q][0]], max_id[tr[q][1]]);
}
----
    r -> 版本tire树
    c -> 传入的值
    l -> 限制条件
int query(int r, int c, int l)
{
    int p = root[r];

    for(int i = 23; i >= 0; i--)
    {
        int v = c >> i & 1;
        if(max_id[tr[p][v ^ 1]] >= l) p = tr[p][v ^ 1];
        else p = tr[p][v];
    }
    return c ^ s[max_id[p]];
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cin>>n>>m;
    string op;
    ----- 
    |tire 头|
    max_id[0] = -1;
    root[0] = ++idx;
    insert(0, 23, 0, root[0]);
    -----
    for(int i = 1; i <= n; i++)
    {
        cin>>a[i];
        s[i] = s[i - 1] ^ a[i];
        root[i] = ++idx;
        insert(i, 23, root[i - 1], root[i]);
    }

    while (m--) {
        cin>>op;
        int x, l, r;
        if(op == "A")
        {
            cin>>x;
            n++;
            s[n] = s[n - 1] ^ x;
            root[n] = ++idx;
            insert(n, 23, root[n - 1], root[n]);
        }
        else
        {
            cin>>l>>r>>x;
            cout<<query(r - 1, s[n] ^ x, l - 1)<<'\n';
        }
    }
}
```

可持续化tire

https://www.acwing.com/solution/content/51419/

异或和:

https://www.acwing.com/solution/content/46534/

## 主席树(可持续化数据结构)

### 板子

```
C++
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <vector>
using namespace std;
const int N = 1e5 + 10, M = 1e4 + 10;
int n, m;
int a[N];
// 离散化数组
vector<int> nums;

struct node
{
    int l, r;
    int cnt;
}tr[N * 4 + N * 17];
// root [1, i] 区间的权值线段树根节点
int root[N], idx;

int find(int x)
{
    return lower_bound(nums.begin(), nums.end(), x) - nums.begin();
}

int build(int l, int r)
{
    int p = ++idx;
    if(l == r)
        return p;
    int mid = (l + r) >> 1;
    tr[p].l = build(l, mid), tr[p].r = build(mid + 1, r);
    return p;
}
// 增加一个数 p  为上一个的根节点
int insert(int p, int l, int r, int x)
{
    int q = ++idx;
    tr[q] = tr[p];
    if(l == r)
    {
        tr[q].cnt++;
        return q;
    }
    int mid = (l + r) >> 1;
    if(x <= mid) tr[q].l = insert(tr[p].l, l, mid, x);
    else tr[q].r = insert(tr[p].r, mid + 1, r, x);
    tr[q].cnt = tr[tr[q].l].cnt + tr[tr[q].r].cnt;
    return q;
}
//查询
int query(int q, int p, int l, int r, int k)
{
    if(l == r)
        return r;
    // 对位相减
    int cnt = tr[tr[q].l].cnt - tr[tr[p].l].cnt;
    int mid = (l + r) >> 1;
    if(k <= cnt) return query(tr[q].l, tr[p].l, l, mid, k);
    else return query(tr[q].r, tr[p].r, mid + 1, r, k - cnt);
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);

    cin>>n>>m;
    for(int i = 1; i <= n; i++)
    {
        cin>>a[i];
        nums.push_back(a[i]);
    }
    sort(nums.begin(), nums.end());
    nums.erase(unique(nums.begin(), nums.end()), nums.end());

    root[0] = build(0, nums.size() - 1);

    for(int i = 1; i <= n; i++)
        root[i] = insert(root[i - 1], 0, nums.size() - 1, find(a[i]));

    while(m--)
    {
        int l, r, k;
        cin>>l>>r>>k;
        cout<<nums[query(root[r], root[l - 1], 0, nums.size() - 1, k)]<<'\n';
    }
}
```

主席树:https://www.acwing.com/solution/content/4224/

## 线段树&&树状数组

### 线段树板子(lazytag) 维护 ax + b

https://ac.nowcoder.com/acm/contest/19684/D

```
C++
#include <bits/stdc++.h>
#include <cstdio>
using namespace std;
typedef long long ll;
const int N = 1e5  + 10;

ll a[N];

struct tnode{
    // sum 1 2 // lazy 
    ll x , x_2, a, b;
    int l, r;
};

struct segment_tree
{
    tnode t[4 * N];

    void init_lazy(int root)
    {
        t[root].a = 1;
        t[root].b = 0;
    }

    // 合并 ch 的 lazy tag
    void union_lazy(int fa, int ch)
    {
        ll ta, tb;
        // a = a1 * a2
        ta = t[fa].a * t[ch].a;
        // b = a1 * b2 + b1
        tb = t[fa].a * t[ch].b + t[fa].b;

        t[ch].a = ta;
        t[ch].b = tb;

        return;
    }

    // 更新 root 点的 推标记后的前缀和
    // 添加值受区间长度影响
    void cal_lazy(int root)
    {
        ll x , x_2;
        tnode nnode = t[root];
        x_2 = nnode.a * nnode.a * nnode.x_2 +
              (nnode.r - nnode.l + 1) * nnode.b * nnode.b +
              2 * nnode.a * nnode.b * nnode.x;

        x   = (nnode.r - nnode.l + 1) * nnode.b + nnode.a * nnode.x;

        t[root].x_2 = x_2;
        t[root].x = x;

        return;
    }

    // 下传 lazy tag
    void push_down(int root)
    {
        if (t[root].a != 1 || t[root].b != 0) 
        {
            cal_lazy(root);
            if(t[root].l != t[root].r)
            {
                int ch = root << 1;
                union_lazy(root, ch);
                union_lazy(root, ch + 1);
            }
            init_lazy(root);
        }
    }

    // 更新 节点前缀和
    void update(int root)
    {
        int ch = root << 1;
        push_down(ch);
        push_down(ch + 1);
        t[root].x = t[ch].x + t[ch + 1].x;
        t[root].x_2 = t[ch].x_2 + t[ch + 1].x_2;
    }

    void build(int root, int l, int r)
    {
        t[root].l = l;
        t[root].r = r;
        init_lazy(root);
        if(l != r)
        {
            int mid = (l + r) >> 1;
            int ch = root << 1;
            build(ch, l, mid);
            build(ch + 1, mid + 1, r);
            update(root);
        }
        else
        {
            t[root].x = a[l];
            t[root].x_2 = a[l] * a[l];
        }
    }

    void change(int root, int l, int  r, ll delta, int op)
    {
        if(t[root].r < l || t[root].l > r) return;
        push_down(root);
        if(l == t[root].l && r == t[root].r)
        {
            if(op == 1)
                t[root].a = delta;
            else 
                t[root].b = delta;
            return;
        }
        int mid = (t[root].l + t[root].r) >> 1;
        int ch = root << 1;
        if(r <= mid)
            change(ch, l, r, delta, op);
        else if(l > mid)
            change(ch + 1, l, r, delta, op);
        else {
            change(ch, l, mid, delta, op);
            change(ch + 1, mid + 1, r, delta, op);
        }
        update(root);
    }

    ll sum(int root, int l, int r, int op)
    {
        push_down(root);
        if(t[root].l == l && t[root].r == r)
        {
            if(op == 1)
                return t[root].x;
            else
                return t[root].x_2;
        }

        int mid = (t[root].l + t[root].r) >> 1;
        int ch = root << 1;
        if(r <= mid)
            return sum(ch, l, r, op);
        else if(l > mid)
            return sum(ch + 1, l, r, op);
        else
            return sum(ch, l, mid, op) + sum(ch + 1, mid + 1, r, op);
    }
};

segment_tree t;
int n, m, op, l, r;
ll x;
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cin>>n>>m;
    for(int i = 1; i <= n; i++)
        cin>>a[i];

    t.build(1, 1, n);

    for(int i = 1; i <= m; i++)
    {
        cin>>op>>l>>r;
        switch (op) 
        {
            case 1:
                cout<<t.sum(1, l, r, 1)<<endl;
                break;
            case 2:
                cout<<t.sum(1, l, r, 2)<<endl;
                break;
            case 3:
                cin>>x;
                t.change(1, l, r, x, 1);
                break;
            case 4:
                cin>>x;
                t.change(1, l, r, x, 2);
                break;
        }
    }
}
```

### 平方序列

> 给 [ l , r ] 区间加 (平方数 数组)
>
> 推公式 父节点~子节点的合并因子

https://ac.nowcoder.com/acm/contest/19684/G

```
C++
#include <bits/stdc++.h>
using namespace std;
const int N = 2e6 + 10;
typedef long long ll;
const ll mod = 1e9 + 7;

struct tnode{
    int l, r;
    ll ci, h, hh;
    ll i, ii, xi;
};
tnode operator + (tnode a, tnode b)
{
    (a.h = a.h + b.h) %= mod;
    (a.ci = a.ci + b.ci) %= mod; 
    (a.hh = a.hh + b.hh) %= mod;
    return a;
}

struct segment_tree{
    tnode t[N * 4];

    void lazy_init(int root)
    {
        t[root].ci = t[root].h = t[root].hh = 0;
    }

    void lazy_union(int fa, int ch)
    {
        t[ch] = t[ch] + t[fa];
    }

    void cal_lazy(int root)
    {
        t[root].xi = t[root].xi % mod 
            + t[root].ci * t[root].ii % mod 
            - 2 * t[root].i * t[root].h % mod
            + 2 * mod 
            + t[root].hh * (t[root].r - t[root].l + 1) % mod;
        t[root].xi %= mod;
    }

    void pushdown(int root)
    {
        if(t[root].ci != 0)
        {
            cal_lazy(root);
                int ch = root << 1;
                lazy_union(root, ch);
                lazy_union(root, ch | 1);
            lazy_init(root);
        }
    }

    void update(int root)
    {
        int ch = root << 1;
        pushdown(ch);
        pushdown(ch | 1);
        t[root].i = t[ch].i + t[ch | 1].i;
        t[root].i %= mod;
        t[root].ii = t[ch].ii + t[ch | 1].ii;
        t[root].ii %= mod;
        t[root].xi = t[ch].xi + t[ch | 1].xi;
        t[root].xi %= mod;
    }

    void build(int root, int l, int r)
    {
        t[root].l = l, t[root].r = r;
        lazy_init(root);
        if(l != r)
        {
            int ch = root << 1;
            int mid = l + ((r - l) >> 1);
            build(ch, l, mid);
            build(ch | 1, mid + 1, r);
            update(root);
        }
        else
        {
            t[root].i = l % mod;
            t[root].ii = 1ll * l * l % mod;
        }
    }

    void change(int root, int l, int r, ll val)
    {
        pushdown(root);
        if(t[root].l == l && t[root].r == r)
        {
            t[root].ci = 1 % mod, t[root].h = val % mod, t[root].hh = val * val % mod; 
            return;
        }
        int mid = t[root].l + ((t[root].r - t[root].l) >> 1);
        /* int mid = (t[root].l + t[root].r) >> 1; */
        int ch = root << 1;
        if(r <= mid)
        {
            change(ch, l, r, val);
        }
        else if(l > mid)
        {
            change(ch | 1, l, r, val);
        }
        else
        {
            change(ch, l, mid, val);
            change(ch | 1, mid + 1, r, val);
        }
        update(root);
    }

    ll query(int root, int l, int r)
    {
        pushdown(root);
        if(l <= t[root].l && r >= t[root].r)
            return t[root].xi % mod;
        int mid = t[root].l + ((t[root].r - t[root].l) >> 1);
        /* int mid = (t[root].l + t[root].r) >> 1; */
        int ch = root << 1;
        if(r <= mid)
            return query(ch, l, r) % mod;
        else if(l > mid)
            return query(ch | 1, l, r) % mod;
        else
            return query(ch, l, mid) % mod + query(ch | 1, mid + 1, r) % mod;
    }
};

segment_tree ts;

int n,m;
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);

    cin>>n>>m;
    ts.build(1, 1, n);
    while (m--) 
    {
        int op, l, r;
        cin>>op>>l>>r;
        if(op == 1)
            ts.change(1, l, r, l - 1);
        else
            cout<<ts.query(1, l, r) % mod<<'\n';
    }
}
```

### 线段树二分查找(仓鼠的鸡蛋)

> 一排桶, 每个桶有一样容量上限, n 组鸡蛋(vi), 找到最靠左的 桶 且有足够空间 放入

https://ac.nowcoder.com/acm/contest/19684/H

```
C++
#include <cstdio>
#include <iostream>
using namespace std;
const int N = 3e5 +10;
int cnt[N];
int t, n, m, k;
struct node{
    int l, r;
    // 维护区间的已经放入容量最大值
    int qmax;
};
int a[N];

struct segment_tree{
    node t[N * 4];

    void update(int root)
    {
        int ch = root << 1;
        t[root].qmax = max(t[ch].qmax, t[ch + 1].qmax);
    }

    void build(int root, int l, int r)
    {
        t[root].l = l, t[root].r = r;
        if(l != r)
        {
            int ch = root << 1;
            int mid = l + ((r - l) >> 1);
            build(ch, l, mid);
            build(ch + 1, mid + 1, r);
            update(root);
        }
        else
        {
            t[root].qmax = m; 
            cnt[l] = 0;
        }
    }

    // 二分修改 偏左端
    int change(int val)
    {
        int root = 1;
        while(t[root].l != t[root].r)
        {
            if(t[root << 1].qmax >= val)
                root <<= 1;
            else
                root = (root << 1) + 1;
        }
        t[root].qmax -= val;
        ++cnt[t[root].l];
        int ans = t[root].l;
        if(k == cnt[t[root].l])
        {
            t[root].qmax = 0;
        }

        root >>= 1;
        while (root)
        {
            update(root);
            root >>= 1;
        }
        return ans;
    }
};


segment_tree ts;

int main()
{
    scanf("%d",&t);
    while (t--) {
        scanf("%d%d%d", &n, &m, &k);
        ts.build(1, 1, n);
        int val;
        for(int i = 1; i <= n; i++)
        {
            scanf("%d",&val);
            if(val > m)
                puts("-1");
            else
                printf("%d\n",ts.change(val));
        }
    }
}
```

### 真二分线段树(针对智乃线段树做了 查询操作)

> 大致题意
> 给定一个整数n, 表示有n个花瓶(初始为空花瓶), 编号从0~n-1. 有如下两种操作:
>
> ①从编号为x的花瓶开始, 要放y朵花, 从前往后一次遍历, 如果是空花瓶则放一朵花在里面, 直至放完所有花或者遍历到最后一个花瓶为止. 倘若此时还有花放不下, 则将它们直接丢弃.
> ②清理[l, r]区间的所有花瓶, 如果里面有花则将其丢弃
>
> 对于每个操作①, 需要输出第一个放花的位置和最后一个放花的位置. 倘若一朵花都放不下, 需要输出”Can not put any one.”
> 对于每个操作②, 需要输出该区间被清理的花的数量
>
> 解题思路
> 线段树的区间修改和区间查询.
>
> 线段树维护当前区间内可以放花朵的数目.
>
> 解法一:
> 对于每个操作①而言, 首先用全局变量L, R维护第一个放花与最后一个放花的位置.
> 找到符合要求的区间后, 如果当前的区间可以放的花朵数量<=需要放置, 则将当前区间放满花朵, 同时查看当前区间第一个和最后一个空花瓶位置, 并更新L与R. 反之若当前区间不满足要求应当先递归左子树, 再递归右子树(因为要尽可能把前面的空花瓶都放满).
> 上述思路我们需要实现两个操作: 找到当前区间第一个空花瓶与最后一个空花瓶. 我们可以分别写两个函数来实现.
>
> 对于每个操作②而言, 就是简单的区间查询, 我们可以顺便进行花瓶清空操作.
>
> 解法二:
> 首先说操作②的实现: 我们的查询函数会返回[l, r]区间的空花瓶数目, 则操作②清理花的数目= [l, r]长度 - [l, r]空花瓶个数.
>
> 对于操作①: 我们其实可以通过查询[x, n]区间空花瓶的个数, 来判断是否一朵花都放不下. 如果可以放的下花的话, 我们可以采用二分的思路去二分[x, n]区间, 来得到第num个空花瓶所在的位置.
>
> 特别注意: 对于最后放花的位置: 如果可以放得下所有y朵花, 我们应查询第y个空花瓶的位置, 反之我们应当查询[x, n]区间最后一个空花瓶所在的位置
>
> [https://blog.csdn.net/weixin_45799835/article/details/110137033?ops_request_misc=&request_id=&biz_id=102&utm_term=Vases%20and%20Flowers&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-3-110137033.pc_search_result_hbase_insert&spm=1018.2226.3001.4187](https://blog.csdn.net/weixin_45799835/article/details/110137033?ops_request_misc=&request_id=&biz_id=102&utm_term=Vases and Flowers&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-3-110137033.pc_search_result_hbase_insert&spm=1018.2226.3001.4187)

```
C++
#include <algorithm>
#include <cstdio>
#include <iostream>
using namespace std;
const int N = 1e5 + 10;
int n, m;
// 全局变量
int al, ar;
struct node{
    int l, r;
    int cou, lazy;
};

struct sg{
    node t[N << 2];

    void lazy_init(int root)
    {
        t[root].lazy = -1;
    }

    void union_lazy(int fa, int ch)
    {
        t[ch].lazy = t[fa].lazy;
    }

    void cal_lazy(int root)
    {
        t[root].cou = t[root].lazy * (t[root].r - t[root].l + 1);
    }

    void pushdown(int root)
    {
        if(t[root].lazy != -1)
        {
            cal_lazy(root);
            if(t[root].l != t[root].r)
            {
                int ch = root << 1;
                // 下传完后 字节点 要更新
                union_lazy(root, ch);
                cal_lazy(ch);

                union_lazy(root, ch | 1);
                cal_lazy(ch | 1);
            }
            lazy_init(root);
        }
    }

    // 同样功能的 lazy 下传
    void pd(node& op, int lazy)
    {
        op.cou = lazy * (op.r - op.l + 1);
        op.lazy = lazy;
    }

    void pd(int x)
    {
        if(t[x].lazy == -1) return;
        int ch = x << 1;
        pd(t[ch], t[x].lazy), pd(t[ch | 1], t[x].lazy);
        t[x].lazy = -1;
    }

    void update(int root)
    {
        int ch = root << 1;
        pushdown(ch), pushdown(ch | 1);
        t[root].cou = t[ch].cou + t[ch | 1].cou;
    }

    void build(int root, int l, int r)
    {
        t[root] = {l, r, 1, -1};
        if(l != r)
        {
            int ch = root << 1;
            int mid = (l + r) >> 1;
            build(ch, l, mid);
            build(ch | 1, mid + 1, r);
            update(root);
        }
    }
    //// 二分 /////
    // 找出最左侧可以放花位置
    int searchl(int root)
    {
        if (t[root].l == t[root].r)
            return t[root].l;
        pushdown(root);
        /* pd(root); */
        int ch = root << 1;
        if(t[ch].cou)
            return searchl(ch);
        else
            return searchl(ch | 1);
    }

    // 找出最右侧可以放花位置
    int searchr(int root)
    {
        if(t[root].l == t[root].r)
            return t[root].l;
        pushdown(root);
        /* pd(root); */
        int ch = root << 1;
        if(t[ch | 1].cou)
            return searchr(ch | 1);
        else
            return searchr(ch);
    }

    void change(int root, int l, int r, int& cot)
    {
        if(!cot || !t[root].cou) return;
        if(t[root].l >= l && t[root].r <= r)
        {
            if(t[root].cou <= cot)
            {
                al = min(al, searchl(root)), ar = max(ar, searchr(root));
                cot -= t[root].cou;
                /* pushdown(root); */
                // change lazy!!!
                pd(t[root], 0);
                return;
            }
            if(t[root].l == t[root].r) return;
        }
        pushdown(root);
        int mid = (t[root].l + t[root].r) >> 1;
        int ch = root << 1;
        if(r <= mid)
            change(ch, l, r, cot);
        else if(l > mid)
            change(ch | 1, l, r, cot);
        else
            change(ch, l, mid, cot), change(ch | 1, mid + 1, r, cot);
        update(root);

        /* pd(root); */
        /* int mid = (t[root].l + t[root].r) >> 1; */
        /* int ch = root << 1; */
        /* if(l <= mid) change(ch, l, r, cot); */
        /* if(r > mid) change(ch | 1, l, r, cot); */
        /* update(root); */
    }

    int query(int root, int l, int r)
    {
        if(t[root].l >= l && t[root].r <= r)
        {
            int res = t[root].r - t[root].l + 1 - t[root].cou;
            /* pushdown(root); */
            /* pd(root); */
            // change lazy !!!
            pd(t[root], 1);
            return res;
        }
        pushdown(root);
        /* pd(root); */
        int mid = (t[root].l + t[root].r) >> 1;
        int ch = root << 1;

        // 不行
        /* if(r <= mid) */
        /*     return query(ch, l, r); */
        /* else if(l > mid) */
        /*     return query(ch | 1, l, r); */
        /* else */ 
        /*     return query(ch, l, mid) + query(ch | 1, mid + 1, r); */

        // 改版后支持查询后懒标记更新 update
        int res = 0;
        if(r <= mid)
            res = query(ch, l, r);
        else if(l > mid)
            res = query(ch | 1, l, r);
        else
            res = query(ch , l, mid) + query(ch | 1, mid + 1, r);
        update(root);
        return res;

        // 查询过程需要 调用lazy tag :
        /* int res = 0; */
        /* if(l <= mid) */
        /*     res += query(ch, l, r); */
        /* if(r > mid) */
        /*     res += query(ch | 1, l, r); */
        /* update(root); */
        /* return res; */
    }
};

sg tre;

void solve()
{
    cin>>n>>m;
    tre.build(1, 1, n);
    while (m--)
    {
        int op, a, b;
        cin>>op>>a>>b;
        if(op == 1)
        {
            al = n, ar = 0;
            int tmp = b;
            tre.change(1, a + 1, n, b);
            if(b != tmp)
                cout<<al - 1<<' '<<ar - 1<<'\n';
            else
                cout<<"Can not put any one.\n";
        }
        else
            cout<<tre.query(1, a + 1, b + 1)<<'\n';
    }
    
}


int main()
{
    ios::sync_with_stdio(false);cin.tie(0);cout.tie(0); 
    int T;
    cin>>T;
    /* scanf("%lld", &T); */
    while(T--)
    {
        solve();
        cout<<'\n';
    }
}
```

### 区间排序(分裂合并树)

二分验证答案+离线配合+线段树维护区间01串1的个数+思维转化(多次操作 一次查询)

https://www.luogu.com.cn/problem/P2824

```
C++
#include <iostream>
#include <algorithm>
using namespace std;
typedef long long ll;
const int N = 1e6 + 10;
int n, m, k;
int a[N], b[N], c[N];
struct tnode{
    int l, r;
    // 记录区间 1 的个数
    int sum;
    // 记录区间覆盖情况 1 or 0
    int tag;
};
/* 统计区间01串的 1 的个数 */
/* 修改01排序 */
struct segment{
    tnode t[N << 2];


    /* lazy当sum副本 */
    /* 故lazy不存在对当前节点影响 */

    void push_tag(int root, int num) 
    {
        t[root].sum = (t[root].r - t[root].l + 1) * num;
        t[root].tag = num;
    }

    void pushdown(int root)
    {
        int ch = root << 1;
        if(t[root].tag != -1)
        {
            push_tag(ch, t[root].tag);
            push_tag(ch | 1, t[root].tag);
            t[root].tag = -1;
        }
    }

    void update(int root)
    {
        int ch = root << 1;
        t[root].sum = t[ch].sum + t[ch | 1].sum;
        if(t[root].l == t[root].r)
            return;
    }

    void build(int root, int l,int r)
    {
        t[root].l = l, t[root].r = r, t[root].sum = 0;
        t[root].tag = -1;
        if(l != r)
        {
            int ch = root << 1;
            int mid = (l + r) >> 1; 
            build(ch, l, mid);
            build(ch | 1, mid + 1, r);
            update(root);
        }
        else
        {
            t[root].sum = c[l];
        }
    }

    // 手动选修改区间
    void change(int root, int l, int r, int num)
    {
        //玄学出界re !!!
        if(t[root].l > r || t[root].r < l) return;
        pushdown(root);
        /* if(t[root].l > r || t[root].r < l) */
        /*     return; */
        if(t[root].l >= l && t[root].r <= r)
        {
            push_tag(root, num);
            return;
        }
        // 手动区分修改区间
        int mid = (t[root].l + t[root].r) >> 1;
        int ch = root << 1;
        if(r <= mid)
            change(ch, l, r, num);
        else if(l > mid)
            change(ch | 1, l, r, num);
        else
        {
            change(ch, l, mid, num);
            change(ch | 1, mid + 1, r, num);
        }
        update(root);
    }

    void _change(int root, int l, int r, int num)
    {
        // 出界
        if(t[root].l > r || t[root].r < l)
            return;
        if(t[root].l >= l && t[root].r <= r)
        {
            push_tag(root, num);
            return;
        }
        pushdown(root);
        int ch = root << 1;
        // 直接下找
        change(ch, l, r, num);
        change(ch | 1, l, r, num);
        update(root);
    }

    int query(int root, int l, int r)
    {
        if(t[root].l > r || t[root].r < l)
            return 0;
        if(t[root].l >= l && t[root].r <= r)
            return t[root].sum;
        /* int mid = (t[root].l + t[root].r) >> 1; */
        int ch = root << 1;
        pushdown(root);
        /* if(r <= mid) */
        /*     return query(ch, l, r); */
        /* else if(l > mid) */
        /*     return query(ch | 1, l, r); */
        /* else */
        /*     return query(ch, l, mid) + query(ch | 1, mid + 1, r); */
        return query(ch, l, r) + query(ch | 1, l, r);
    }
};

segment tre;
struct qry{
    int op, l, r;
};
qry q[N];
/* m   是查找的位置 */
/* ans 是二分的测试答案 */
/* 一种经典的二分状态设置，大于等于x设置为1，小于x设置为0 */
/* 大于等于ans -> 1 */
/* 小于    ans -> 0 */
bool check(int ans)
{
    for(int i = 1; i <= n; i++)
    {
        if(a[i] >= ans)
            c[i] = 1;
        else 
            c[i] = 0;
    }
    int s = 0;
    tre.build(1, 1, n);
    for(int i = 1; i <= m; i++)
    {
        s = tre.query(1, q[i].l, q[i].r);
        if(q[i].op)
        {
            tre.change(1, q[i].l, q[i].l + s - 1, 1);
            tre.change(1, q[i].l + s, q[i].r, 0);
        }
        else
        {
            tre.change(1, q[i].l, q[i].r - s, 0);
            tre.change(1, q[i].r - s + 1, q[i].r, 1);
        }
        /* for(int i = 1; i <= n; i++) */
        /*     cout<<c[i]<<' '; */
        /* cout<<ans<<';'<<s<<endl; */
    }
    s = tre.query(1, k, k);
    return s == 1;
}

void solve()
{
    for(int i = 1; i <= n; i++)
    {
        cin>>a[i];
        b[i] = a[i];
    }

    for(int i = 1; i <= m; i++)
    {
        int op, l, r;
        cin>>op>>l>>r;
        q[i] = {op, l, r};
    }
    sort(b + 1, b + n + 1);
    cin>>k;
    int l = 1, r = n;
    while(l < r)
    {
        int mid = (l + r + 1) >> 1;
        /* m 位置上为 1 == 该位置正确数 比 mid 大 */
        if(check(mid))
            l = mid;
        else
            r = mid - 1;
    }
    cout<<b[l]<<'\n';
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cin>>n>>m;
    solve();
}
```

### 扫描线

#### 模板

https://www.luogu.com.cn/problem/P5490

[![imaged818935608ff1757.png](https://kyros.oss-cn-shenzhen.aliyuncs.com/markdown/imaged818935608ff1757.png?x-oss-process=style/kyros)](https://kyros.oss-cn-shenzhen.aliyuncs.com/markdown/imaged818935608ff1757.png?x-oss-process=style/kyros)

```
C++
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
typedef long long ll;
const int N = 1e5 + 10;

vector<int> ns;
// 离散化 find函数
ll f(ll x)
{
    return lower_bound(ns.begin(), ns.end(), x) - ns.begin(); 
}
// 获得 离散前的坐标值
ll g(ll x)
{
    return ns[x];
}

int n;
ll x1, y1, x2, y2;
// 扫描线
struct scanline{
    // x1 x2 横线段覆盖
    // h 到时候算高度差的
    ll x1, x2, h;
    // 权值 入边 or 出边
    int mark;
    // 后面 sort 排序用
    bool operator < (const scanline& s) const{
        return h < s.h;
    }
};
scanline line[N << 2];

struct node{
    ll l, r;
    // len 区间覆盖长度 
    // sum 区间出现的次数
    ll sum, len;
};

struct segment_tree{
    node t[N << 3];

    // ch && ch | 1 合并信息 到 root
    void update(int root)
    {
        int ch = root << 1;
        if(t[root].sum)
            t[root].len = g(t[root].r + 1) - g(t[root].l);
        else if(t[root].l != t[root].r)
            t[root].len = t[ch].len + t[ch | 1].len;
        else
            t[root].len = 0;
    }

    // 建树
    void build(int root, int l, int r)
    {
        t[root] = {l , r, 0, 0};
        if(l != r)
        {
            int mid = (l + r) >> 1;
            int ch = root << 1;
            build(ch, l, mid), build(ch | 1, mid + 1, r);
            update(root);
        }
    }


    // 修改
    void change(int root, int l, int r, int k)
    {
        if(t[root].l >= l && t[root].r <= r)
        {
            t[root].sum += k;
            update(root);
        }
        else
        {
            int mid = (t[root].l + t[root].r) >> 1;
            int ch = root << 1;
            // 自动进入判越界:
            if(l <= mid)
                change(ch, l, r, k);
            if(r > mid)
                change(ch | 1, l, r, k);
            // 手动区分修改区间: (递归次数会少)
            /* if(r <= mid) */
            /*     change(ch, l, r, k); */
            /* else if(l > mid) */
            /*     change(ch | 1, l, r, k); */
            /* else */
            /* { */
            /*     change(ch, l, mid, k); */
            /*     change(ch | 1, mid + 1, r, k); */
            /* } */
            
            update(root);
        }
    }
};


segment_tree tree;
void solve()
{
    cin>>n;
   
    for(int i = 1, cnt = 0; i <= n; i++)
    {
        cin>>x1>>y1>>x2>>y2;
        // 长方形 左下点 & 右上点
        line[cnt++] = {x1, x2, y1, 1};
        line[cnt++] = {x1, x2, y2, -1};
        ns.push_back(x1), ns.push_back(x2);
    }

    // 离散
    sort(ns.begin(), ns.end());
    ns.erase(unique(ns.begin(), ns.end()), ns.end());
    sort(line , line + n * 2);
    
    tree.build(1, 0, ns.size() - 2);

    ll ans = 0;
    for(int i = 0; i < n * 2; i++)
    {
        // t[1] 是 当前 覆盖的 有效长度
        // 两次 扫描线 的 长度
        // 扫过的面积 get
        if(i > 0)
            ans += tree.t[1].len * (line[i].h - line[i - 1].h);
        // 
        tree.change(1, f(line[i].x1), f(line[i].x2) - 1, line[i].mark);
    }
    cout<<ans<<'\n';
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    solve();
}
```

#### 周长并

https://vjudge.net/problem/POJ-1177

```
C++
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
using namespace std;
const int N = 1e5 + 10;
int n;
int X[N << 1];
vector<int> ns;

int find(int x)
{
    return lower_bound(ns.begin(), ns.end(), x) - ns.begin() + 1;
}

int get(int x)
{
    return ns[x - 1];
}

struct scanline{
    int l, r, h;
    int mark;
    bool operator < (const scanline& s) const{
        if(h == s.h)
            return mark < s.mark;
        return h < s.h;
    }
}line[N];

struct tnode{
    // sum 被覆盖的次数, len 区间有效线段长度
    // c 表示区间线段的个数
    int l, r, sum, len, c;
    // 左端点, 右端点 是否被 覆盖
    bool lc, rc;
};

struct seg{
    tnode t[N << 2];

    void update(int root)
    {
        // 如果被覆盖的区间
        if(t[root].sum)
        {
            // len
            /* t[root].len = get(t[root].r + 1) - get(t[root].l); */
            t[root].len = X[t[root].r + 1] - X[t[root].l];  
            // 左右端点 至少覆盖了 1 次
            t[root].lc = t[root].rc = 1;
            // 标记段
            t[root].c = 1;
        }
        else
        {
            int ch = root << 1;
            t[root].len = t[ch].len + t[ch | 1].len;
            // 左右端点标记 上浮
            t[root].lc = t[ch].lc, t[root].rc = t[ch | 1].rc;
            // 统计 段数
            t[root].c = t[ch].c + t[ch | 1].c;

            // 中间是连续的一段
            if(t[ch].rc && t[ch | 1].lc) t[root].c -= 1;
        }
    }

    void build(int root, int l, int r)
    {
        t[root] = {l, r, 0, 0, 0, 0, 0};
        int ch = root << 1;
        int mid = (l + r) >> 1;
        if(l != r)
        {
            build(ch, l, mid);
            build(ch | 1, mid + 1, r);
        }
    }

    void change(int root, int l, int r, int c)
    {
        // 
        if(t[root].l >= find(l) && t[root].r <= find(r) + 1)
        {
            t[root].sum += c;
            update(root);
            return;
        }
        int ch = root << 1;
        int mid = (t[root].l + t[root].r) >> 1;
        if(r <= mid)
            change(ch, l, r, c);
        else if(l > mid)
            change(ch | 1, l, r, c);
        else 
            change(ch, l, mid, c), change(ch | 1, mid + 1, r, c);
        update(root);
    }

    void _change(int root, int l, int r, int c)
    {
        int tl = t[root].l, tr = t[root].r;
        if(X[tl] >= r || X[tr + 1] <= l)
            return;
        if(l <= X[tl] && X[tr + 1] <= r)
        {
            t[root].sum += c;
            update(root);
            return;
        }
        int ch = root << 1;
        _change(ch, l, r, c);
        _change(ch | 1, l, r, c);
        update(root);
        /* int mid = (t[root].l + t[root].r) >> 1; */
        /* if(r <= mid) */
        /*     _change(ch, l, r, c); */
        /* else if(l > mid) */
        /*     _change(ch | 1, l, r, c); */
        /* else */
        /*     _change(ch, l, mid, c), _change(ch | 1, mid + 1, r, c); */
        /* update(root); */
    }
};

seg tre;
void solve()
{
    cin>>n;
    for(int i = 1; i <= n; i++)
    {
        int x1, y1, x2, y2;
        cin>>x1>>y1>>x2>>y2;
        line[i * 2 - 1] = {x1, x2, y1, 1};
        line[i * 2] = {x1, x2, y2, -1};
        /* ns.push_back(x1), ns.push_back(x2); */
        X[i * 2 - 1] = x1, X[i * 2] = x2;
    }
    n <<= 1;
    sort(line + 1, line + n + 1);

    sort(X + 1, X + n + 1);
    int tot = unique(X + 1, X + n + 1) - X - 1;
    tre.build(1, 1, tot - 1);

    /* sort(ns.begin(), ns.end() - 1); */
    /* ns.erase(unique(ns.begin(), ns.end()), ns.end()); */
    /* tre.build(1, 1, ns.size()); */
    int res = 0;
    int pre = 0;

    for(int i = 1; i < n; i++)
    {
        /* tre.change(1, line[i].l, line[i].r + 1, line[i].mark); */
        tre._change(1, line[i].l, line[i].r, line[i].mark);

        res += abs(pre - tre.t[1].len);

        pre = tre.t[1].len;

        res += 2 * tre.t[1].c * (line[i + 1].h - line[i].h);
    }

    res += line[n].r - line[n].l;

    cout<<res<<endl;
}

int main()
{
    solve();
}
```

#### 区间离散点权值(lazytag new 玩法)

> 拓展点成框解决 选中权值最大问题

https://www.luogu.com.cn/problem/P1502

[![星星2](https://kyros.oss-cn-shenzhen.aliyuncs.com/markdown/p295zpbp.png?x-oss-process=style/kyros)](https://kyros.oss-cn-shenzhen.aliyuncs.com/markdown/p295zpbp.png?x-oss-process=style/kyros)

```
C++
#include <functional>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;
typedef long long  ll;
const int N = 1e6 + 10;
int x, y, w, h, a;
int n;
vector<int> p;
struct scanline{
    ll x1, x2, h;
    ll mark;
    bool operator < (const scanline& t) const{
        return (h == t.h ? mark > t.mark : h < t.h);
    }
};

scanline line[N << 1];

struct tnode{
    int l, r;
    // 记录当前权值 mx  tag: add (暂存副本 如果 再次更新往下再下推)
    ll mx, add;
};

ll find(ll x)
{
    return lower_bound(p.begin(), p.end(), x) - p.begin();
}

struct seg_tree{
    tnode t[N << 3];

    void lazy_init(int root)
    {
        t[root].add = 0;
    }

    void union_lazy(int fa, int ch)
    {
        t[ch].add += t[fa].add;
    }

    //  标记 不影响 节点值 -- > 只用于 查询 提速 
    //void cal_lazy(int root)
    //{
    //    t[root].mx += t[root].add;
    //}

    /* ???? */ 
    /* 先把 lazytag 加到子节点的maxsum里 */
    // add 懒标记作为sum 的 副本
    /* 再下传父节点的 lazytag */
    /* ???? */
    void pushdown(int root)
    {
        int ch = root << 1;
        if(t[root].add)
        {
            /* cal_lazy(root); */
            t[ch].mx += t[root].add;
            t[ch | 1].mx += t[root].add;
            if(t[root].l != t[root].r)
            {
                union_lazy(root, ch);
                union_lazy(root, ch | 1);
            }
            lazy_init(root);
        }
    }

    void update(int root)
    {
        int ch = root << 1;
        /* pushdown(ch); */
        /* pushdown(ch | 1); */
        t[root].mx = max(t[ch].mx, t[ch | 1].mx);
    }

    void build(int root, int l, int r)
    {
        t[root] = {l, r, 0, 0};
        if(l != r)
        {
            int mid = (l + r) >> 1;
            int ch = root << 1;
            build(ch, l, mid);
            build(ch | 1, mid + 1, r);
        }
    }

    void change(int root, int l, int r, int k)
    {
        if(t[root].l >= l && t[root].r <= r)
        {
            t[root].mx += k;
            t[root].add += k;
            return;
        }
        else
        {
            pushdown(root);
            int mid = (t[root].l + t[root].r) >> 1;
            int ch = root << 1;
            /* if(l <= mid) */
            /*     change(ch, l, r, k); */
            /* if(r > mid) */
            /*     change(ch | 1, l, r, k); */
            if(r <= mid)
                change(ch, l, r, k);
            else if(l > mid)
                change(ch | 1, l, r, k);
            else 
            {
                change(ch, l, mid, k);
                change(ch | 1, mid + 1, r, k);
            }
            update(root);
        }
    }
};

seg_tree tre;
void solve()
{
    cin>>n>>w>>h;
    p.clear();
    /* memset(tre.t, 0, sizeof(tre.t)); */
    /* memset(line, 0, sizeof(line)); */
    for(int i = 1, cnt = 0; i <= n; i++)
    {
        cin>>x>>y>>a;
        line[cnt++] = {x, x + w - 1, y, a};
        line[cnt++] = {x, x + w - 1, y + h - 1, -a};
        p.push_back(x), p.push_back(x + w - 1);
    }
    sort(p.begin(), p.end());
    sort(line, line + 2 * n);
    p.erase(unique(p.begin(), p.end()), p.end());


    tre.build(1, 1, p.size());
    ll ans = 0;
    for(int i = 0; i <= 2 * n; i++)
    {
        tre.change(1, find(line[i].x1), find(line[i].x2), line[i].mark);
        ans = max(tre.t[1].mx, ans);
    }
    cout<<ans<<endl;
}

int main()
{
    int T;
    cin>>T;
    while (T--) {
        solve();
    }
}
```

### 维护区间最长子序列

#### 区间最长连续子段和(有负数)

```
C++
#include <iostream>
#include <algorithm>
#include <type_traits>
using namespace std;
const int N = 1e6 + 10;
int n, m;
int a[N];
struct node {
    int l, r;
    int sum, lmax, rmax, tmax;
};

struct seg{
    node t[N << 2];
    int leaf[N];

    void pushup(node &u, node &l, node &r) {
        // 最大连续字段和: 在左子 ; 在右子 ; 横跨左右
        u.tmax = max(l.tmax, max(r.tmax, r.lmax + l.rmax));
        // 前缀和: 左边的最大前缀和 ; 左边和总和+右边的最大前缀和
        u.lmax = max(l.lmax, l.sum + r.lmax);
        // 后缀和: 右边的最大后缀和 ; 右边的总和+左边的最大后缀和
        u.rmax = max(r.rmax, r.sum + l.rmax);
        u.sum = l.sum + r.sum;
    }

    void pushup(int u)
    {
        pushup(t[u], t[u<<1], t[u<<1|1]);
    }
    void update(int root)
    {
        int ch = root << 1;
        t[root].sum = t[ch].sum + t[ch | 1].sum;
        t[root].lmax = max(t[ch].sum + t[ch | 1].lmax, t[ch].lmax);
        t[root].rmax = max(t[ch | 1].sum + t[ch].rmax, t[ch | 1].rmax);
        t[root].tmax = max(max(t[ch].tmax, t[ch | 1].tmax), t[ch].rmax + t[ch | 1].lmax);
    }

    void cg(int root, int val)
    {
        t[root].lmax = t[root].rmax = t[root].sum = t[root].tmax = val;
    }

    void build(int root, int l, int r)
    {
        t[root] = {l, r};
        if(l != r)
        {
            int mid = (l + r) >> 1;
            int ch = root << 1;
            build(ch, l, mid);
            build(ch | 1, mid + 1, r);
            /* update(root); */
            pushup(root);
        }
        else
        {
            cg(root, a[l]);
            leaf[l] = root;
        }
    }

    void change(int root, int x, int v)
    {
        if(t[root].l == x && t[root].r == x)
        {
            /* cg(root, v); */
            t[root] = {x, x, v, v, v, v};
            return;
        }
        else
        {
            int mid = (t[root].l + t[root].r) >> 1;
            int ch = root << 1;
            if(x <= mid)
                change(ch, x, v);
            else
                change(ch | 1, x, v);
            /* update(root); */
            pushup(root);
        }
    }

    // 不行的 点修改方式
    /* void change(int pos, int k) */
    /* { */
    /*     int root = leaf[pos]; */
    /*     cg(root, k); */
    /*     root >>= 1; */
    /*     while(root) */
    /*     { */
    /*         update(root); */
    /*         root >>= 1; */
    /*     } */
    /*     update(root); */
    /* } */

    node query(int root, int l, int r) {
        if (t[root].l >= l && t[root].r <= r)
            return t[root];
        int ch = root << 1;
        int mid = (t[root].l + t[root].r) >> 1;
        if (r <= mid)
            return query(ch, l, r);
        else if (l > mid)
            return query(ch | 1, l, r);
        else {
            node nd1 = query(ch, l, mid);
            node nd2 = query(ch | 1, mid + 1, r);
            /* int tmax = max(max(nd1.tmax, nd2.tmax), nd1.rmax + nd2.lmax); */
            /* node end = {0, 0, 0, 0, 0, tmax}; */
            /* return end; */
            node node;
            pushup(node, nd1, nd2);
            return node;
        }
    }

    // 拿出节点 在算合并
    node _query(int u, int l, int r) {
        if (t[u].l >= l && t[u].r <= r) {
            return t[u];
        }else{
            int mid=(t[u].l+t[u].r)>>1;
            if(r<=mid)return query(u<<1, l, r);
            else if(mid<l)return query(u<<1|1, l, r);
            else{
                auto tr1=query(u<<1, l, r);
                auto tr2=query(u<<1|1,l,r);
                node node;
                pushup(node,tr1,tr2);
                return node;
            }
        }
    }
};

seg tre;
int main()
{
    cin>>n>>m;
    for(int i = 1; i <= n; i++)
        cin>>a[i];
    tre.build(1, 1, n);
    while(m--)
    {
        int op, x, y;
        cin >> op >> x >> y;
        if (op == 1) {
            if (x > y)
                swap(x, y);
            cout << tre.query(1, x, y).tmax << '\n';
        }
        else
            /* tre.change(x, y); */
            tre.change(1, x, y);
    }
}
```

#### 区间最长01串

>  第一种思路：利用线段树求解， 线段树维护一个最长连续子段和，然后查询的时候判断x点是在左边区间还是右边区间，如果在左边区间就判断左边区间的rmax(包含右断点的最大连续子段和)的长度是否超过了x那个点，如果超过了那么直接返回 左边区间.rmax + 右边区间.lmax。 否则继续查询左边区间。x在右边区间同理。
> ​ 第二种思路：利用树状数组 + 二分。 每次二分搜索左边和右边连续的最后一个1的位置。2个位置相减就是连续的子段和了。
> ​ 第三种思路：利用set自带的红黑树。 如第二种思路，我们只需要求出左边右边第一个0的位置（毁坏的村庄）,那么中间的必然是好的村庄。

##### 线段树做法:

```
C++
#include <algorithm>
#include <cstdio>
#include <iostream>
using namespace std;
const int N = 1e5 + 10;
typedef long long ll;
int n, m;
struct node{
    int l, r;
    ll lmax, rmax;
    ll tmax;
    /* ll sum; */
};

struct seg{
    node t[N << 1];
    int leaf[N];

    void update(int root)
    {
        int ch = root << 1;
        int mid = (t[root].l + t[root].r) >> 1;
        t[root].lmax = t[ch].lmax + (mid - t[root].l + 1 == t[ch].lmax ? t[ch | 1].lmax : 0);
        t[root].rmax = t[ch | 1].rmax + (t[root].r - mid == t[ch | 1].rmax ? t[ch].rmax : 0); 
        t[root].tmax = max(max(t[ch].tmax, t[ch | 1].tmax), t[ch].rmax + t[ch | 1].lmax);
        /* t[root].sum = t[ch].sum + t[ch | 1].sum; */
    }

    void update(int root, int val)
    {
        t[root].lmax = t[root].rmax = t[root].tmax = val;
    }

    void build(int root, int l, int r)
    {
        t[root] = {l, r};
        if(l != r)
        {
            int ch = root << 1;
            int mid = (l + r) >> 1;
            build(ch, l, mid);
            build(ch | 1, mid + 1, r);
            update(root);
        }
        else
        {
            update(root, 1);
            leaf[l] = root;
        }
    }

    void change(int root, int pos, int val)
    {
        if(t[root].l == pos && t[root].r == pos)
        {
            update(root, val);
            return;
        }
        int ch = root << 1;
        int mid = (t[root].l + t[root].r) >> 1;
        if(pos <= mid)
            change(ch, pos, val);
        else
            change(ch | 1, pos, val);
        update(root);
    }


    int query(int root, int pos)
    {
        if(t[root].l == t[root].r)
            return t[root].tmax;
        int mid = (t[root].l + t[root].r) >> 1;
        int ch = root << 1;
        if(pos <= mid)
        {
            if(mid - pos + 1 <= t[ch].rmax)
                return t[ch].rmax + t[ch | 1].lmax;
            else
                return query(ch, pos);
        }
        else
        {
            if(pos - mid <= t[ch | 1].lmax)
                return t[ch].rmax + t[ch | 1].lmax;
            else
                return query(ch | 1, pos);
        }
    }
};

seg tre;
int ret[N];
void solve()
{
    tre.build(1, 1, n);
    int last = 0;
    while(m --)
    {
        char op[2];
        int x;
        scanf("%s", op);
        /* cin>>op; */
        if(*op == 'D')
        {
            scanf("%d", &x);
            ret[++last] = x;
            tre.change(1, x, 0);
        }
        else if(*op == 'R')
            tre.change(1, ret[last--], 1);
        else
        {
            scanf("%d", &x);
            printf("%d\n", tre.query(1, x));
        }
    }
}

int main()
{
    while(~scanf("%d%d", &n, &m))
    {
        solve();
    }
}
```

##### 树状数组+二分查找

```
C++
#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;
const int N = 1e5 + 10;
int t[N], ret[N], last;
int n, m;
bool st[N];
int lowbit(int x)
{
    return x & -x;
}

void add(int x, int val)
{
    for(int i = x; i <= n; i += lowbit(i))
        t[i] += val;
}

int qry(int x)
{
    int ans = 0;
    for(int i = x; i; i -= lowbit(i))
        ans += t[i];
    return ans;
}

int searchl(int x, int l, int r)
{
    int num = qry(x);
    while(l < r)
    {
        int mid = (l + r) >> 1;
        int t = num;
        if(mid != 1) t -= qry(mid - 1);
        if(t == (x - mid + 1))
            r = mid;
        else
            l = mid + 1;
    }
    return l;
}

int searchr(int x, int l, int r)
{
    int num = qry(x);
    while(l < r)
    {
        int mid = (l + r + 1) >> 1;
        int t = qry(mid) - num;
        if(t == (mid - x))
            l = mid;
        else
            r = mid - 1;
    }
    return r;
}

void solve()
{
    for(int i = 1; i <= n; i++)
        add(i, 1);
    char op[2];
    last = 0;
    int x;
    for(int i = 1; i <= m; i++)
    {
        scanf("%s", op);
        if(op[0] == 'D')
        {
            scanf("%d", &x);
            ret[++last] = x;
            if(st[x]) add(x, -1), st[x] = false;
        }
        else if(op[0] == 'R')
        {
            /* add(ret[last--], 1); */
            if(st[ret[last]])
                last--;
            else
                st[ret[last]] = true, add(ret[last--], 1);
        }
        else
        {
            scanf("%d", &x);
            int t = qry(x);
            if(x != 1) t -= qry(x - 1);
            if(t == 0)
                puts("0");
            else
            {
                int l = searchl(x, 1, x);
                int r = searchr(x, x, n);
                printf("%d\n", r - l + 1);
            }
        }
    }
}

int main()
{
    while(~scanf("%d%d", &n, &m))
    {
        memset(t, 0, sizeof t);
        memset(st, true, sizeof st);
        solve();
    }
}
```

##### stl (set 的 红黑树)

```
C++
#include <cstdio>
#include <iostream>
#include <set>
using namespace std;
const int N = 1e5 + 10;
int n, m, rec[N], last, x;
char op[2];

void solve()
{
    last = 0;
    set<int> st;
    for(int i = 1; i <= m; i++)
    {
        scanf("%s", op);
        if(op[0] == 'D')
        {
            scanf("%d", &x);
            st.insert(x);
            rec[++last] = x;
        }
        else if(op[0] == 'Q')
        {
            scanf("%d", &x);
            /* 查找 set 中 大于等于 x 的 第一个位置 作为 r 小于 x 的 第一个 作为 l */
            auto t = st.lower_bound(x);
            int r = (t == st.end() ? n + 1 : *t);
            int l = (t == st.begin() ? 0 : *(--t));
            if(r == x)
                puts("0");
            else
                /* r - l + 1 - 2 (左右两个 d 节点) */
                printf("%d\n", r - l - 1);
        }
        else
            st.erase(rec[last--]);
    }
}

int main()
{
    while(~scanf("%d%d", &n, &m))
    {
        solve();
    }
}
```

### 多个懒标记(有优先覆盖顺序)维护最长1串

> 题目大意：查找是否有连续的空余时间
>
> 他的操作有三个 ：
>
> ds x 查找屌丝的时间有连续的x长度的空余时间
>
> ns x 为女神安排时间，先查找屌丝时间是否有空余，没有的话无视屌丝，直接判断女神时间是否有连续x长的空余时间
>
> stduy a b 清空a到b区间的事情

lazy分开封装处理

```
C++
#include <algorithm>
#include <cstdio>
#include <iostream>
using namespace std;
const int N = 1e5 + 10;
int n, m;
struct node{
    int l, r;
    int d, n, s;
    int dls, drs, dsm;
    int nls, nrs, nsm;
};

struct seg{
    node t[N << 3];
    // lots of lazy tag zone

    void male(int root)
    {
        node& rt = t[root];
        rt.d = 1;
        rt.dls = rt.drs = rt.dsm = 0;
    }

    void female(int root)
    {
        node& rt = t[root];
        rt.n = 1;
        rt.d = 0;
        rt.nls = rt.nrs = rt.nsm = 0;
        rt.dls = rt.drs = rt.dsm = 0;
    }

    void learn(int root)
    {
        node& rt = t[root];
        rt.s = 1, rt.d = rt.n = 0;
        rt.dls = rt.drs = rt.dsm = (rt.r - rt.l + 1);
        rt.nls = rt.nrs = rt.nsm = (rt.r - rt.l + 1);
    }

    // lazy tag init in the progress
    /* void lazy_init(int root) */
    /* { */
    /*     node& rt = t[root]; */
    /*     rt.d = rt.n = rt.s = 0; */
    /* } */
    
    /* void union_lazy(int fa, int ch) */
    /* { */
    /*     t[ch].d = t[fa].d; */
    /*     t[ch].n = t[fa].n; */
    /*     t[ch].s = t[fa].s; */
    /* } */
    // 延迟标记下传递
    void pushdown(int root)
    {
        node& rt = t[root];
        int ch = root << 1;
        if(rt.s)
        {
            learn(ch);
            learn(ch | 1);
            rt.s = 0;
        }
        if(rt.d)
        {
            male(ch);
            male(ch | 1);
            rt.d = 0;
        }
        if(rt.n)
        {
            female(ch);
            female(ch | 1);
            rt.n = 0;
        }
        /* update(root); */
    }
    /* void pushdown(int root) */
    /* { */
    /*     node& rt = t[root]; */
    /*     if(rt.s || rt.n || rt.d) */
    /*     { */
    /*         int ch = root << 1; */
    /*         if(rt.l != rt.r) */
    /*         { */
    /*             union_lazy(root, ch); */
    /*             cal_lazy(ch); */
    /*             union_lazy(root, ch | 1); */
    /*             cal_lazy(ch | 1); */
    /*         } */
    /*         lazy_init(root); */
    /*     } */
    /* } */

    void update(int root)
    {
        int ch = root << 1;
        int mid = (t[root].l + t[root].r) >> 1;
        t[root].dsm =
            max(max(t[ch].dsm, t[ch | 1].dsm), t[ch].drs + t[ch | 1].dls);
        t[root].nsm =
            max(max(t[ch].nsm, t[ch | 1].nsm), t[ch].nrs + t[ch | 1].nls);

        t[root].dls = 
            t[ch].dls + (mid - t[root].l + 1 == t[ch].dls ? t[ch | 1].dls : 0);
        t[root].nls = 
            t[ch].nls + (mid - t[root].l + 1 == t[ch].nls ? t[ch | 1].nls : 0);

        t[root].drs =
            t[ch | 1].drs + (t[root].r - mid == t[ch | 1].drs ? t[ch].drs : 0);
        t[root].nrs = 
            t[ch | 1].nrs + (t[root].r - mid == t[ch | 1].nrs ? t[ch].nrs : 0);
    }

    void build(int root, int l, int r)
    {
        t[root] = {l, r, 0, 0, 0, 1, 1, 1, 1, 1, 1};
        /* lazy_init(root); */
        if(l != r)
        {
            int ch = root << 1;
            int mid = (l + r) >> 1;
            build(ch, l, mid);
            build(ch | 1, mid + 1, r);
            update(root);
        }
    }
    void insert(int root, int l, int r, int flag)
    {
        node& rt = t[root];
        if(rt.l == l && rt.r == r)
        {
            if(flag == 1)
                male(root);
            else
                female(root);
            return;
        }
        int mid = (rt.l + rt.r) >> 1;
        int ch = root << 1;
        pushdown(root);
        if(r <= mid)
            insert(ch, l, r, flag);
        else if(l > mid)
            insert(ch | 1, l, r, flag);
        else
        {
            insert(ch, l, mid, flag);
            insert(ch | 1, mid + 1, r, flag);
        }
        update(root);
    }
    // highest level -- learn

    void study(int root, int l, int r)
    {
        node& rt = t[root];
        if(rt.l == l && rt.r == r)
        {
            learn(root);
            return;
        }
        int mid = (rt.l + rt.r) >> 1;
        int ch = root << 1;
        pushdown(root);
        if(r <= mid)
            study(ch, l, r);
        else if(l > mid)
            study(ch | 1, l, r);
        else 
        {
            study(ch, l, mid);
            study(ch | 1, mid + 1, r);
        }
        update(root);
    }

    // 全区间查询
    int query(int root, int flag, int cnt)
    {
        int ch = root << 1;
        int mid = (t[root].l + t[root].r) >> 1;
        if(t[root].l == t[root].r)
            return t[root].l;
        pushdown(root);
        if(flag == 1)
        {
            if (t[ch].dsm >= cnt)
                return query(ch, flag, cnt);
            else if (t[ch].drs + t[ch | 1].dls >= cnt)
                return mid - t[ch].drs + 1;
            else
                return query(ch | 1, flag, cnt);
        }
        else 
        {
            if(t[ch].nsm >= cnt)
                return query(ch, flag, cnt);
            else if(t[ch].nrs + t[ch | 1].nls >= cnt)
                return mid - t[ch].nrs + 1;
            else
                return query(ch | 1, flag, cnt);
        }
    }
};
seg tre;
int cnt;
int main()
{
    int T;
    scanf("%d", &T);
    char str[20];
    while(T--)
    {
        scanf("%d%d", &n, &m);
        printf("Case %d:\n", ++cnt);
        tre.build(1, 1, n);
        /* tre.study(1, 1, n); */
        while(m--)
        {
            int x, y;
            int pos = 0;
            scanf("%s", str);
            if(str[0] == 'D')
            {
                scanf("%d", &x);
                if(tre.t[1].dsm < x)
                    printf("fly with yourself\n");
                else
                {
                    pos = tre.query(1, 1, x);
                    tre.insert(1, pos, pos + x - 1, 1);
                    printf("%d,let's fly\n", pos);
                }
            }
            else if(str[0] == 'N')
            { 
                scanf("%d", &x);
                if(tre.t[1].dsm < x)
                {
                    if(tre.t[1].nsm < x)
                        printf("wait for me\n");
                    else
                    {
                        pos = tre.query(1, 2, x);
                        tre.insert(1, pos, pos + x - 1, 2);
                        printf("%d,don't put my gezi\n", pos);
                    }
                }
                else
                {
                    pos = tre.query(1, 1, x);
                    tre.insert(1, pos, pos + x - 1, 2);
                    printf("%d,don't put my gezi\n", pos);
                }
            }
            else
            {
                scanf("%d%d", &x, &y);
                tre.study(1, x, y);
                printf("I am the hope of chinese chengxuyuan!!\n");
            }
        }
    }
}
```

### 树套树(可以转为主席树)

https://ac.nowcoder.com/acm/contest/19684/F

[![image-20211027211407492](https://kyros.oss-cn-shenzhen.aliyuncs.com/markdown/image-20211027211407492.png?x-oss-process=style/kyros)](https://kyros.oss-cn-shenzhen.aliyuncs.com/markdown/image-20211027211407492.png?x-oss-process=style/kyros)

一个线段树求当前维护最小lastpos(x)

lastpos(x) -> 这个x 最后出现的位置pos

得出个MEX

一个树状数组求 区间里面 比 MEX 大的数多少个 == len - 比MEX小的数多少个

离线操作 q1 q2

q1 {id, l, r}

q2 {id, pos, MEX, type} 2 * m 个

(type +- 1 -> 区间求和 pre_arr)

妙点:

a[i] 1e9 离散(pb(x), pb(x + 1)) or 吧大于n的数变为n + 1(迟早都得改的数)

牛客入门题:

```
C++
#include <algorithm>
#include <iostream>
#include <type_traits>
#include <vector>
using namespace std;
const int N = 3e5 + 10;
typedef long long ll;
int n, m;

struct tnode{
    int l, r;
    int mpos;
};

struct bnode{
    int l, r;
    int cnt;
};

struct segment{
    tnode t[N << 2];
    int leaf[N];

    void update(int root)
    {
        int ch = root << 1;
        t[root].mpos = min(t[ch].mpos, t[ch | 1].mpos);
    }

    void build(int root, int l, int r)
    {
        t[root].l = l, t[root].r = r;
        if(l != r)
        {
            int mid = (l + r) >> 1;
            int ch = root << 1;
            build(ch, l, mid);
            build(ch | 1, mid + 1, r);
        }
        else
        {
            t[root].mpos = 0;
            leaf[l] = root;
        }
    }

    void change(int x, int pos)
    {
        /* while(t[root].l != t[root].r) */
        /* { */
        /*     int mid = (t[root].l + t[root].r) >> 1; */
        /*     if(x > mid) */
        /*         root = root << 1 | 1; */
        /*     else */
        /*         root <<= 1; */
        /* } */
        int root = leaf[x];
        t[root].mpos = pos;
        root >>= 1;
        while (root) 
        {
            update(root);
            root >>= 1;
        }
    }

    int query(int l)
    {
        int root = 1;
        while(t[root].l != t[root].r)
        {
            int ch = root << 1;
            if(l > t[ch].mpos)
                root <<= 1;
            else
                root = root << 1 | 1;
        }
            return t[root].l - 1;
    }
};

/* TODO: 统计个数的线段树 */
/* struct bsegment{ */
/*     bnode t[N << 2]; */
/*     int leaf[N]; */

/*     void update(int root) */
/*     { */
/*         int ch = root << 1; */
/*         t[root].cnt = t[ch].cnt + t[ch | 1].cnt; */
/*     } */

/*     void build(int root, int l, int r) */
/*     { */
/*         t[root] = {l, r, 0}; */
/*         if(l != r) */
/*         { */
/*             int mid = (l + r) >> 1; */
/*             int ch = root << 1; */
/*             build(ch, l, mid); */
/*             build(ch | 1, mid + 1, r); */
/*             update(root); */
/*         } */
/*     } */
    
/*     void change(int pos) */
/*     { */
/*         int root = leaf[pos]; */
/*         t[root].cnt++; */
/*     } */
/* }; */

int bt[N];
int a[N];
vector<int> p;

ll lowbit(ll x)
{
    return x & -x;
}

void add(ll x)
{
    for(ll i = x; i <= n; i += lowbit(i))
        bt[i]++;
}

ll sum(ll x)
{
    ll s = 0;
    for(ll i = x; i; i -= lowbit(i))
        s += bt[i];
    return s;
}

segment tre;

struct q1_node{
    int id;
    int l, r;
}q1[N << 1];

bool cmp1(const q1_node& x, const q1_node& y)
{
    return x.r < y.r;
}

struct q2_node{
    int id;
    int type;
    int num;
    int pos;
}q2[N << 1];

int ans[N];
bool cmp2(const q2_node& x, const q2_node& y)
{
    return x.pos < y.pos;
}

void solve()
{
    cin>>n;
    for(int i = 1; i <= n; i++)
    {
        cin>>a[i];
        if(a[i] > n) a[i] = n + 1;
        bt[i] = 0; 
    }
    cin>>m;
    for(int i = 1; i <= m; i++)
    {
        int l, r;
        cin>>l>>r;
        q1[i] = {i, l, r};
        ans[i] = r - l + 1;
    }
    sort(q1 + 1, q1 + m + 1, cmp1);
    int now = 1;
    int qtot = 0;
    tre.build(1, 1, n + 1);
    for(int i = 1; i <= n; i++)
    {
        tre.change(a[i], i);
        while(now <= m && q1[now].r == i)
        {
            int q1_as = tre.query(q1[now].l);
            q2[++qtot].id = q1[now].id;
            q2[qtot].num = q1_as;
            q2[qtot].pos = q1[now].r;
            q2[qtot].type = -1;
            if(q1[now].l != 1)
            {
                q2[++qtot].id = q1[now].id;
                q2[qtot].num = q1_as;
                q2[qtot].pos = q1[now].l - 1;
                q2[qtot].type = 1;
            }
            ++now;
        }
    }
    sort(q2 + 1, q2 + 1 + qtot, cmp2);
    now = 1;
    for(int i = 1; i <= n; i++)
    {
        add(a[i]);
        while(now <= qtot && q2[now].pos == i)
        {
            ans[q2[now].id] += q2[now].type * sum(q2[now].num); 
            ++now;
        }
    }
    for(int i = 1; i <= m; i++)
    {
        cout<<ans[i]<<'\n';
    }
}



int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    int T = 1;
    while (T--)
    {
        solve();
    }
}
```

## st/rmq/跳表

(好写 快!) 区间最大值, 可以往后插入log(n)

```
C++
#include <cstdio>
#include <iostream>
#include <cmath>
#include <math.h>
using namespace std;
typedef long long ll;
const int N = 2e6 + 10;
const int M = 21;

int n, m;
int dp[N][M];
int w[N];
ll D;
int cnt;

// 尾插一个新数 log(n)
void change(int pos, int val)
{
    dp[pos][0] = val;
    for(int i = 1; pos - (1 << i) >= 0; i++)
        dp[pos][i] = max(dp[pos][i - 1], dp[pos - (1 << (i - 1))][i - 1]);
}

void init()
{
    for(int k = 0; k < M; k++)
    {
        for(int i = 1; i + (1 << k) - 1 <= n; i++)
        {
            if(!k)
                dp[i][k] = w[i];
            else
                dp[i][k] = max(dp[i][k - 1], dp[i + (1 << (k - 1))][k - 1]);
        }
    }
}

void pf()
{
    for(int i = 0; i <= cnt; i++)
    {
        for(int k =0; k < M; k++)
            cout<<dp[i][k]<<' ';
        cout<<endl;
    }
    cout<<endl;
}

int query(int l, int r)
{
    int len = r - l + 1;
    int k = log(len) / log(2);
    // 前往后
    return max(dp[l][k], dp[r - (1 << k) + 1][k]);
    // 后往前
    return max(dp[r][k], dp[l + (1 << k) - 1][k]);
}
int main()
{
    cin>>n>>D;
    char op[2];
    int t = 0;
    while (n--) {
        scanf("%s", op);
        ll x;
        if(*op == 'A')
        {
            cin>>x;
            ll tmp = (x + t) % D; 
            change(++cnt, tmp);
        }
        else
        {
            /* pf(); */
            cin>>x;
            if(x == 1)
            {
                t = dp[cnt][0];
                cout<<t<<'\n';
                continue;
            }
            t = query(cnt - x + 1, cnt);
            cout<<t<<'\n';
        }
    }
}
```

### 统计区间数字频率

> 对上升序列如：1 1 2 2 2 3 3 4 5 5 ……. 统计区间出现次数最多数个数。
>
> 我们可以构造一个b[]数组，
>
> if（a[i]==a[i-1]）b[i]=b[i-1]+1;
>
> else b[i]=1;
>
> 这样对上述例子，b[]数组有1 2 1 2 3 1 2 1 1 2
>
> 那么对询问区间[l,r]，如果l在数与数交界处，那么直接查询l,r区间最大值。
>
> 否则要知道与a[l]相同延伸到end，那么这个区间大小end-l+1，与rmq(end+1，r)取最大值就是答案。

```
C++
#include <algorithm>
#include <cstring>
#include <iostream>
#include <cmath>
#include <math.h>
using namespace std;
const int N = 1e6 + 10;
const int M = 21;
int n, m;
int w[N];
int dp[N][M];

inline int read()
{
    int f=1,p=0;char c=getchar();
    while(c>'9'||c<'0'){if(c=='-')f=-1;c=getchar();}
    while(c>='0'&&c<='9'){p=p*10+c-'0';c=getchar();}
    return f*p;
}
inline void init()
{
    /* memset(dp, -1, sizeof(-1)); */
    for(int k = 0; k < M; k++)
        for(int i = 1; i + (1 << k) - 1 <= n; i++)
            if(!k)
                dp[i][k] = w[i];
            else
                dp[i][k] = max(dp[i][k - 1], dp[i + (1 << (k - 1))][k - 1]);
}

inline int query(int l, int r)
{
    if(l > r) return 0;
    int len = r - l + 1;
    int k = log(1.0 * len) / log(2.0);
    return max(dp[l][k], dp[r - (1 << k) + 1][k]);
}
int num[N];

inline void solve()
{
    m = read();
    for(int i = 1; i <= n; i++)
        num[i] = read();

    w[1] = 1;
    for(int i = 2; i <= n; i++)
    {
        if(num[i] == num[i - 1])
            w[i] = w[i - 1] + 1;
        else
            w[i] = 1;
    }
    init();
    for(int i = 1; i <= m; i++)
    {
        int l, r;
        l = read(), r = read();
        int t = l;
        while(t <= r && num[t] == num[t - 1])t++;
        cout<<max(query(t, r), t - l)<<'\n';
    }
    /* for(int i = 1; i <= n; i++) */
    /* { */
    /*     cout<<w[i]<<' '; */
    /* } */
}

int main()
{
    while (cin>>n && n) {
        solve();
    }
}
```

### 降雨量

```
C++
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <utility>
#include <map>
#include <vector>
#include <cmath>
#include <math.h>
using namespace std;
typedef pair<int, int> PII;
typedef long long  ll;
const int N = 5e3 + 10;
const int M = 20;
ll n, m;
ll dp[N][M];
ll w[N];

inline int gi(){
    int a=0;char x=getchar();bool f=0;
    while((x<'0'||x>'9')&&x!='-')x=getchar();
    if(x=='-')x=getchar(),f=1;
    while(x>='0'&&x<='9')a=(a<<3)+(a<<1)+x-48,x=getchar();
    return f?-a:a;
}
void init()
{
    for(int k = 0; k < M; k++)
    {
        for(int i = 1; i + (1 << k) - 1 <= n; i++)
        {
            if(!k)
                dp[i][k] = w[i];
            else
                dp[i][k] = max(dp[i][k - 1], dp[i + (1 << (k - 1))][k - 1]);
        }
    }
}

int query(int l, int r)
{
    if(l > r)
        return 0;
    ll len = r - l + 1;
    ll k = log(1.0 * len) / log(2.0);
    return max(dp[l][k], dp[r - (1 << k) + 1][k]);
}

map<int, int>mp;
vector<int> years(N, 0);
int main()
{
    n = gi();
    for(int i = 1; i <= n; i++)
    {
        ll year, data;
        year = gi(), data = gi();
        mp[year] = i;
        years[i] = year;
        w[i] = data;
    }
    init();
    m = gi();

    /* for(auto t : mp) */
    /* { */
    /*     cout<<t.first<<' '<<w[t.second]<<endl; */
    /* } */
    /* need :: l < x < r */
               /* qry */
    for(int i = 1; i <= m; i++)
    {
        int l, r;
        l = gi(), r = gi();
        if(!mp.count(l) && !mp.count(r))
            puts("maybe");
            /* puts("none has"); */
        else if(!mp.count(l) && mp[r])
        {
            /* puts("hsnt l"); */
            ll pos = lower_bound(years.begin(), years.end(), l) - years.begin() + 1;
            ll qry = query(pos, mp[r] - 1);
            if(qry >= w[mp[r]])
                puts("false");
            else
                puts("maybe");
        }
        else if(mp[l] && !mp.count(r))
        {
            /* puts("hsnt r"); */
            ll pos = lower_bound(years.begin(), years.end(), r) - years.begin() + 1;
            ll qry = query(mp[l] + 1, pos - 1);
            if(qry <= w[mp[l]])
                puts("false");
            else
                puts("maybe");
        }
        else
        {
            /* puts("both has ::"); */
            ll qry = query(mp[l] + 1, mp[r] - 1);
            /* cout<<qry<<endl; */
            if(qry >= w[mp[r]] || qry >= w[mp[l]])
                puts("false");
            else if(mp[r] - mp[l] != r - l)
                puts("maybe");
            else
                puts("true");
        }
    }
}
```

## 分块

### 区间求和 (lazy tag)

```
C++
#include <cstdio>
#include <iostream>
#include <cmath>
using namespace std;
typedef long long ll;
const int N = 1e5 + 10, M = 350;
int w[N];
int n, m, len;
// len 为 sqrt(n)
ll sum[M], add[M];
// 分块
int get(int x)
{
    return x / len;
}

void change(int l, int r, ll d)
{
    // 段内
    if(get(l) == get(r))
        for(int i = l; i <= r; i++)
        {
            w[i] += d;
            sum[get(i)] += d;
        }
    else
    // 两不整段 + 整段
    {
        int i = l, k = r;
        while(get(i) == get(l)) w[i] += d, sum[get(i)] += d, i++;
        while(get(k) == get(r)) w[k] += d, sum[get(k)] += d, k--; 
        for(int q = get(i); q <= get(k); q++) sum[q] += len * d, add[q] += d;
    }
}

ll query(int l, int r)
{
    ll res = 0;
    if(get(l) == get(r))
        for(int i = l; i <= r; i++) res += w[i] + add[get(i)];
    else
    {
        int i = l, k = r;
        while(get(i) == get(l)) res += w[i] + add[get(i)], i++;
        while(get(k) == get(r)) res += w[k] + add[get(k)], k--;
        for(int q = get(i); q <= get(k); q++) res += sum[q];
    }
    return res;
}
```

### 区间开关问题(鸽一鸽)

```
C++
#include <iostream>
#include <cmath>
using namespace std;
const int N = 2e5 + 10;
const int M = 600;
bool w[N], sum[M], add[M];
int n, m, len;

int get(int x)
{
    return x / len;
}
void change(int l, int r)
{
    if(get(l) == get(r))
    {
        int cnt = 0;
        for(int i = l; i <= r; i++)
        {
            w[i] = !w[i];
            if(w[i]) cnt++;
        }
        sum[get(l)] -= (r - l + 1);
        sum[get(l)] += 2 * cnt;
    }
    else
    {
        int tl = l, tr = r;
        int lcnt = 0, rcnt = 0;
        while(get(l) == get(tl)) {
            w[tl] = !w[tl];
            if(w[tl])
                lcnt++;
            tl++;
        }
        sum[get(l)] -= (tl - l + 1);
        sum[get(l)] += 2 * lcnt;

        while (get(r) == get(tr)) {
            w[tr] = !w[tr];
            if(w[tr])
                rcnt++;
            tr--;
        }
        sum[get(r)] -= (r - tr +1);
        sum[get(r)] += 2 * rcnt;

        for(int k = get(tl); k <= get(tr); k++) {
            sum[k] = (len - sum[k]);
            add[k] =! add[k];
        }
    }
}
int query(int l, int r)
{
    int res = 0;
    if(get(l) == get(r))
        for(int i = l; i <= r; i++)
            res += w[i] ^ add[get(i)];
    else
    {
        int tl = l, tr = r;
        while(get(tl) == get(l))
            res += (w[tl] ^ add[get(tl)]), tl++; 
        while(get(tr) == get(r))
            res += (w[tr] ^ add[get(tr)]), tr--;
        for(int k = get(tl); k <= get(tr); k++)
            res += sum[k];
    }
    return res;
}
int main()
{
    cin>>n>>m;
    int op, l, r;
    len = sqrt(n);
    while (m--) {
        cin>>op>>l>>r;
        if(!op)
            change(l, r);
        else
            cout<<query(l, r)<<'\n';
    }
}
```

## 13年杭州签到题(摆烂不会)

```
C++
#include <cstdio>
#include <iostream>
using namespace std;
typedef long long ll;
const int N = 1e5 + 10;
const ll mod = 10007;
int n, m;
struct tnode{
    int l, r;
    ll sumx, sumx2, sumx3;
    ll add, mul, cover;
};

struct seg{
    tnode t[N << 3];

    void lazy_init(int root)
    {
        t[root].add = 0;
        t[root].mul = 1;
        t[root].cover = 0;
    }

    void union_lazy(int fa, int ch)
    {
        /* if(t[fa].cover) */
        /*     t[ch].add = 0, t[ch].mul = 1, t[ch].cover = t[fa].cover; */
        /* t[ch].mul *= t[fa].mul; */
        /* t[ch].mul %= mod; */

        /* t[ch].add = t[fa].mul * t[ch].add % mod + t[fa].add; */
        /* t[ch].add %= mod; */
        t[ch].add = t[fa].mul * t[ch].add % mod + t[fa].add;
        t[ch].add %= mod;
        t[ch].mul = t[fa].mul * t[ch].mul % mod;
        t[ch].mul %= mod;
        t[ch].cover = t[fa].cover;
    }

    void cal_lazy(int root)
    {
        int tx, tx2, tx3;
        tx = t[root].sumx, tx2 = t[root].sumx2, tx3 = t[root].sumx3;

        if(t[root].cover)
        {
            ll len = t[root].r - t[root].l + 1;

            tx = t[root].cover * len;
            tx %= mod;
            
            tx2 = t[root].cover * t[root].cover % mod * len % mod;
            tx2 %= mod;

            tx3 = t[root].cover * t[root].cover % mod * t[root].cover % mod * len % mod;
            tx3 %= mod;

            t[root].add = 0;
            t[root].mul = 1;
        }

        if(t[root].mul != 1)
        {
            tx = tx * t[root].mul;
            tx %= mod;
            tx2 = tx3 * t[root].mul % mod * t[root].mul % mod;
            tx2 %= mod;
            tx3 = tx3 * t[root].mul % mod * t[root].mul % mod * t[root].mul % mod;
            tx3 %= mod;
        }

        if(t[root].add)
        {
            tx = t[root].sumx % mod 
                + t[root].add * (t[root].r - t[root].l + 1) % mod;
            tx %= mod;

            tx2 = t[root].sumx2 % mod 
                + 2 * t[root].add % mod * t[root].sumx % mod 
                + (t[root].r - t[root].l + 1) * t[root].add % mod * t[root].add % mod;
            tx2 %= mod;
    
            tx3 = t[root].sumx3 % mod 
                + (t[root].r - t[root].l + 1) * t[root].add % mod * t[root].add % mod * t[root].add % mod
                + 3 * t[root].add % mod * t[root].sumx2 % mod
                + 3 * t[root].sumx % mod * t[root].add % mod * t[root].add % mod;
            tx3 %= mod;
        }
        t[root].sumx = tx;
        t[root].sumx2 = tx2;
        t[root].sumx3 = tx3;
    }

    void pushdown(int root)
    {
        if(t[root].add || t[root].mul != 1 || t[root].cover)
        {
            cal_lazy(root);
            if(t[root].l != t[root].r)
            {
                int ch = root << 1;
                union_lazy(root, ch);
                union_lazy(root, ch | 1);
            }
            lazy_init(root);
        }
    }

    void update(int root)
    {
        int ch = root << 1;
        pushdown(ch);
        pushdown(ch | 1);
        t[root].sumx = t[ch].sumx + t[ch | 1].sumx;
        t[root].sumx %= mod;
        t[root].sumx2 = t[ch].sumx2 + t[ch | 1].sumx2;
        t[root].sumx2 %= mod;
        t[root].sumx3 = t[ch].sumx3 + t[ch | 1].sumx3;
        t[root].sumx3 %= mod;
    }

    void build(int root, int l, int r)
    {
        t[root] = {l, r};
        lazy_init(root);
        if(l != r)
        {
            int mid = (l + r) >> 1;
            int ch = root << 1;
            build(ch, l, mid);
            build(ch | 1, mid + 1, r);
        }
        else
        {
            t[root].sumx = t[root].sumx2 = t[root].sumx3 = 0;
        }
    }

    void change(int root, int l, int r, int op, ll d)
    {
        pushdown(root);
        if(t[root].l >= l && t[root].r <= r)
        {
            if(op == 1)
            {
                t[root].add += d;
                t[root].add %= mod;
                return;
            }
            if(op == 2)
            {
                t[root].mul *= d;
                t[root].mul %= mod;
                return;
            }
            if(op == 3)
            {
                t[root].cover = d;
                return;
            }
        }
        int mid = (t[root].l + t[root].r) >> 1;
        int ch = root << 1;
        if(r <= mid)
            change(ch, l, r, op, d);
        else if(l > mid)
            change(ch | 1, l, r, op, d);
        else
        {
            change(ch, l, mid, op, d);
            change(ch | 1, mid + 1, r, op, d);
        }
        update(root);
    }

    ll query(int root, int l, int r, int p)
    {
        pushdown(root);
        if(t[root].l >= l && t[root].r <= r)
        {
            if(p == 1)
                return t[root].sumx % mod;
            if(p == 2)
                return t[root].sumx2 % mod;
            if(p == 3)
                return t[root].sumx3 % mod;
        }
        int mid = (t[root].l + t[root].r) >> 1;
        int ch = root << 1;
        if(r <= mid)
            return query(ch, l, r, p) % mod;
        else if(l > mid)
            return query(ch | 1, l, r, p) % mod;
        else
            return query(ch, l, mid, p) % mod + query(ch | 1, mid + 1, r, p) % mod;
    }
};

seg tre;

void solve()
{
    tre.build(1, 1, n);
    for(int i = 1; i <= m; i++)
    {
        int op, l, r;
        ll d;
        scanf("%d%d%d%lld", &op, &l, &r, &d);
        if(op != 4)
            tre.change(1, l, r, op, d);
        else
            printf("%lld\n", tre.query(1, l, r, d) % mod);
    }
}

int main()
{
    while(~scanf("%d%d", &n, &m))
    {
        if(n == 0 && m == 0) break;
        solve();
    }
}
```

## 并查集

[![在这里插入图片描述](https://kyros.oss-cn-shenzhen.aliyuncs.com/markdown/20200803171631118.jpg?x-oss-process=style/kyros)](https://kyros.oss-cn-shenzhen.aliyuncs.com/markdown/20200803171631118.jpg?x-oss-process=style/kyros)

### 并查集(拓展域 / 带权 + dp)

https://vjudge.net/problem/POJ-1417

拓展域 / 带权 分成连通块

dp 背包 组合

码量大

#### 拓展域

```
C++
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>
const int Base = 1010;
const int N = 2 * Base;
using namespace std;

int p[N];
int n, m, p1, p2;

int find(int x)
{
    return (p[x] == x ? x : p[x] = find(p[x]));
}

int dp[Base][Base];
bool pre[Base][Base];
int cnt[Base];

#define same first
#define diff second

int bk;
pair<int, int> block[N];

void init()
{
    memset(dp, 0, sizeof dp);
    memset(pre, 0, sizeof pre);
    memset(cnt, 0, sizeof cnt);
    bk = 0;
}

void solve()
{
    n = p1 + p2;
    init();
    for(int i = 1; i <= n * 2; i++) p[i] = i;
    // 分块
    for(int i = 1; i <= m; i++)
    {
        int a, b;
        char type[5];
        scanf("%d%d%s", &a, &b, type);

        int pa = find(a), pb = find(b), pan = find(a + n), pbn = find(b + n);
        if(type[0] == 'n')
        {
            p[pa] = pbn;
            p[pan] = pb;
        }
        else
        {
            p[pa] = pb;
            p[pan] =pbn;
        }
    }

    // 记录块内成员数 && 记录 两个集合 i 的 1 ~ n 根 与  n+1 ~ 2n 根
    for(int i = 1; i <= n; i++)
    {
        int x = find(i);
        if(cnt[x] == 0 && x <= n)
        {
            block[++bk] = {x, find(i + n)};
        }
        cnt[x]++;
    }

    // 01 背包 组合  
    // 要 有解 且 唯一
    // dp 记录组合
    // pre 记录路径
    dp[0][0] = 1;
    for(int i = 1; i <= bk; i++)
    {
        for(int k = 0; k <= p1; k++)
        {
            if(dp[i - 1][k])
            {
                if(k + cnt[block[i].same] <= p1)
                {
                    dp[i][k + cnt[block[i].same]] += dp[i - 1][k];
                    pre[i][k + cnt[block[i].same]] = 1;
                }
                if(k + cnt[block[i].diff] <= p1)
                {
                    dp[i][k + cnt[block[i].diff]] += dp[i - 1][k];
                    pre[i][k + cnt[block[i].diff]] = 0;
                }
            }
        }
    }

    if(dp[bk][p1] != 1)
        puts("no");
    else
    {
        vector<int> pans;
        int ct = p1;
        for(int i = bk; i >= 1; i--)
        {
            if(pre[i][ct])
                pans.push_back(block[i].same), ct -= cnt[block[i].same];
            else
                pans.push_back(block[i].diff), ct -= cnt[block[i].diff];
        }

        vector<int> ans;
        for(int i = 1; i <= n; i++)
        {
            int x = find(i);
            if(find(pans.begin(), pans.end(), x) != pans.end())
                ans.push_back(i);
        }
        for(int i = 0; i < ans.size(); i++)
            printf("%d\n", ans[i]);
        puts("end");
    }
}

int main()
{
    while(~scanf("%d%d%d", &m, &p1, &p2))
    {
        if(!m && !p1 && !p2)
            break;
        solve();
    }
}
```

#### 带权并查集

```
C++
#include <cstdio>
#include <iostream>
#include <cstring>
#include <vector>
using namespace std;
const int N = 2010;
int n, m, p1, p2;
int p[N], d[N];
vector<int> block[N], v1[N], v2[N];
int dp[N][310];
bool ans[N];

void init()
{
    for(int i = 1; i <= N; i++)
        p[i] = i, d[i] = 0, block[i].clear(), v1[i].clear(), v2[i].clear();
    memset(dp, 0, sizeof dp);
    memset(ans, 0, sizeof ans);
    n = p1 + p2;
}

int find(int x)
{
    if(x != p[x])
    {
        int px = find(p[x]);
        d[x] ^= d[p[x]];
        p[x] = px;
    }
    return p[x];
}

void unio(int a, int b, int t)
{
    int pa = find(a), pb = find(b);
    if(pa != pb)
    {
        p[pa] = pb;
        d[pa] = d[a] ^ d[b] ^ t;
    }
}

void solve()
{
    init();
    for(int i = 1; i <= m; i++)
    {
        int a, b;
        char type[5];
        scanf("%d%d%s", &a, &b, type);
        if(type[0] == 'n')
            unio(a, b, 1);
        else
            unio(a, b, 0);
    }

    for(int i = 1; i <= n; i++)
    {
        int x = find(i);
        block[x].push_back(i);
    }

    int tot = 1;
    for(int i = 1; i <= n; i++)
    {
        int sz = block[i].size();
        bool bl = 0;
        for (int k = 0; k < sz; k++) 
        {
            int tmp = block[i][k];
            if(d[tmp] == 0)
                v1[tot].push_back(tmp), bl = 1;
            else
                v2[tot].push_back(tmp), bl = 1;
        }
        if(bl)
            tot++;
    }

    // dp 背包
    dp[0][0] = 1;
    for(int i = 1; i < tot; i++)
    {
        int s1 = v1[i].size(), s2 = v2[i].size();
        for(int k = p1; k >= s1; k--)
            dp[i][k] += dp[i - 1][k - s1];
        for(int k = p1; k >= s2; k--)
            dp[i][k] += dp[i - 1][k - s2];
    }

    if(dp[tot - 1][p1] != 1)
        puts("no");
    else
    {
        int tmp = p1;
        for(int i = tot - 1; i >= 1; i--)
        {
            int s1 = v1[i].size(), s2 = v2[i].size();
            if(dp[i - 1][tmp - s1])
            {
                for(int k = 0; k < s1; k++)
                    ans[v1[i][k]] = 1;
                tmp -= s1;
            }
            else
            {
                for(int k = 0; k < s2; k++)
                    ans[v2[i][k]] = 1;
                tmp -= s2;
            }
        }

        for(int i = 1; i <= n; i++)
            if(ans[i])
                printf("%d\n", i);
        puts("end");
    }
}

int main()
{
    while(~scanf("%d%d%d", &m, &p1, &p2))
    {
        if(!m && !p1 & !p2)
            break;
        solve();
    }
}
```

### 并查集/二叉堆优化

https://vjudge.net/problem/POJ-1456

> 有n件商品，每件商品的利润为pi。销售日期的截止时间为di (deadline).一天只能销售一件物品。问这n件商品的最大利润为多少

用一个pre数组记录该时间点前最近的一个空闲时间点

```
C++
#include <iostream>
#include <algorithm>
#include <cstring>
#include <utility>
using namespace std;
const int N = 1e5 + 10;
int pre[N]; // 压缩找 最近 空闲天
pair<int, int> p[N];
int n;
int find(int x)
{
    return (pre[x] == -1 ? x : pre[x] = find(pre[x]));
}

void solve()
{
    for(int i = 1; i <= n; i++)
    {
        int pf, d;
        cin>>pf>>d;
        p[i] = {pf, d};
    }

    memset(pre, -1, sizeof pre);

    sort(p + 1, p + n + 1);
    int ans = 0;
    for(int i = n; i; i--)
    {
        int t = find(p[i].second);
        if(t > 0)
        {
            ans += p[i].first;
            pre[t] = t - 1;
        }
    }
    cout<<ans<<endl;
}

int main()
{
    while(cin>>n)
        solve();
}
```

### 并查集多权

https://vjudge.net/problem/POJ-1984

> 涉及大量的集合合并与查询的操作，用并查集是很好的选择，因为要保存集合内点之间的关系（距离），所以带权并查集
>
> 这道题巧妙的点在于：设置水平和竖直方向两种权值
>
> 坑点在于：并没有明显指出询问的时间（index）是非减的
>
> 距离具备累加性的关系

[![image-20211102103951371](https://kyros.oss-cn-shenzhen.aliyuncs.com/markdown/image-20211102103951371.png?x-oss-process=style/kyros)](https://kyros.oss-cn-shenzhen.aliyuncs.com/markdown/image-20211102103951371.png?x-oss-process=style/kyros)

```
C++
#include <cstdio>
#include <cstdlib>
#include <iostream>
using namespace std;
const int N = 1e5 + 10;
int p[N], dx[N], dy[N], n, e, m;
struct node{
    int f1, f2, dis;
    char dir;
};
node nd[N];

int find(int x)
{
    if(p[x] != x)
    {
        int px = find(p[x]);
        dx[x] += dx[p[x]];
        dy[x] += dy[p[x]];
        p[x] = px;
    }
    return p[x];
}

void unio(int a, int b, int x, int y)
{
    int pa = find(a), pb = find(b);
    if(pa != pb)
    {
        p[pa] = pb;
        dx[pa] = x + dx[b] - dx[a];
        dy[pa] = y + dy[b] - dy[a];
    }
}

void solve()
{
    scanf("%d%d", &n, &e);
    for(int i= 1; i <= n; i++)
        p[i] = i, dx[i] = dy[i] = 0;
    for(int i = 1; i <= e; i++)
    {
        int a, b, dis;
        char dir;
        scanf("%d%d%d %c", &a, &b, &dis, &dir);
        nd[i] = {a, b, dis, dir};
    }
    scanf("%d", &m);
    int pos = 1;
    for(int i = 1; i <= m; i++)
    {
        int a, b, idx;
        scanf("%d%d%d", &a, &b, &idx);
        for (int t = pos; t <= idx; t++) 
        {
            int x = 0, y = 0;
            if(nd[t].dir == 'N')
                y = nd[t].dis;
            if(nd[t].dir == 'S')
                y = -nd[t].dis;
            if(nd[t].dir == 'E')
                x = nd[t].dis;
            if(nd[t].dir == 'W')
                x = -nd[t].dis;
            unio(nd[t].f1, nd[t].f2, x, y);
        }
        pos = idx + 1;
        int pa = find(a), pb = find(b);
        if(pa != pb)
            puts("-1");
        else
            printf("%d\n", abs(dx[a] - dx[b]) + abs(dy[a] - dy[b]));
    }
}

int main()
{
    solve();
}
```

### 逆向并查集

> 逆向思考的并查集
> 题意：
> 给出power值
> 问题：1 查找某点能到达的power值最大的点
> 2 删除给出的边（在做删除此边的操作之前的查询，此边依旧可连通）
> 若正向思考，则无法删除边
> 所以，是逆向的并查集
> 方法：
> 先把所有的问题存储起来
> 合并除了要删除的边以外！的所有的边—>得出parent树—>以拥有最大power的点为树的根节点
> 倒序遍历问题—>若是查询操作，则直接输出该店
> —>若是删除边操作，则将该边添加！！！回去树中，以使该边存在于前面的查询操作！！！！！！！

```
C++
#include <iostream>
#include <cstring>
#include <map>
#include <utility>
#define INF 0x3f3f3f3f
using namespace std;
typedef pair<int, int> PII;
const int N = 5e5 + 10;
int p[N], val[N];
int n, m, q;
map<PII, int> mp;
int ans[N];
PII ve[N];
bool flag = 0;
void init()
{
    for(int i = 0; i <= n; i++)
        p[i] = i;
}

int find(int x)
{
    return (p[x] == x ? x : p[x] = find(p[x]));
}

void unio(int a, int b)
{
    int pa = find(a), pb = find(b);
    if(pa != pb)
    {
        if(val[pa] > val[pb])
            p[pb] = pa;
        else if(val[pa] < val[pb])
            p[pa] = pb;
        else if(pa < pb)
            p[pb] = pa;
        else 
            p[pa] = pb;
    }
}

struct node{
    char s[15];
    int x, y;
} qry[N];

void solve()
{
    if(flag)
        cout<<endl;
    else
        flag = 1;

    init();
    for(int i = 0; i < n; i++)
        cin>>val[i];

    cin>>m;

    mp.clear();

    for(int i = 0; i < m; i++)
    {
        int a, b;
        cin>>a>>b;
        ve[i].first = a, ve[i].second = b;
    }
    cin>>q;
    for(int i = 1; i <= q; i++)
    {
        cin>>qry[i].s;
        if(qry[i].s[0] == 'q')
            cin>>qry[i].x;
        else
        {
            cin>>qry[i].x>>qry[i].y;
            mp[make_pair(qry[i].x, qry[i].y)] = 1;
            mp[make_pair(qry[i].y, qry[i].x)] = 1;
        }
    }

    for(int i = 0; i < m; i++)
    {
        if(mp[make_pair(ve[i].first, ve[i].second)] == 0)
            unio(ve[i].first, ve[i].second);
    }

    memset(ans, INF, sizeof ans);
    for(int i = q; i >= 1; i--)
    {
        if(qry[i].s[0] == 'd')
            unio(qry[i].x, qry[i].y);
        else
        {
            int px = find(qry[i].x);
            if(val[px] > val[qry[i].x])
                ans[i] = px;
            else
                ans[i] = -1;
        }
    }
    for(int i = 1; i <= q; i++)
        if(ans[i] != INF)
            cout<<ans[i]<<endl;
}

int main()
{
    int T;
    /*scanf("%d", &T);*/
    while(cin>>n)
    {
        solve();
    }
}
```

## 最小生成树

### prim

```
C++
#include <iostream>
#include <cstring>
#include <sched.h>
using namespace std;
const int N = 1e3 + 10;
#define INF 0x3f3f3f3f
int g[N][N];
int st[N], dist[N];
int n;

int prim()
{
    memset(dist, INF, sizeof dist);
    int ans = 0;
    dist[1] = 0;
    for(int i = 0; i < n; i++)
    {
        int t = -1;
        for(int k = 1; k <= n; k++)
            if(!st[k] && (t == -1 || dist[t] > dist[k]))
                t = k;
        if(t && dist[t] == INF) return INF;
        st[t] = 1;
        ans += dist[t];
        for (int k = 1; k <= n; k++)
            dist[k] = min(dist[k], g[t][k]);
    }
    return ans;
}

int main()
{
    cin>>n;
    for(int i = 1; i <= n; i++)
        for(int k = 1; k <= n; k++)
            cin>>g[i][k];
    int t = prim();
    cout<<t<<endl;
}
```

### prim堆优化

```
C++
#include <functional>
#include <iostream>
#include <vector>
#include <cstring>
#include <queue>
using namespace std;
const int N = 2e5 + 10;
#define INF 0x3f3f3f3f
#define PII pair<int, int>
int st[510];
int h[N], e[N], w[N], ne[N], idx;
int n, m;

void add(int a, int b, int c)
{
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx++;
}

int prim()
{
    memset(st, 0, sizeof st);
    int sum = 0, cnt = 0;
    priority_queue<PII, vector<PII>, greater<PII>> q;
    q.push({0, 1});

    while (!q.empty()) 
    {
        auto t = q.top();
        q.pop();
        int ver = t.second, dst = t.first;
        if(st[ver]) continue;
        st[ver] = 1, sum += dst, ++cnt;

        for(int i = h[ver]; i != -1; i = ne[i])
        {
            int k = e[i];
            if(!st[k])
                q.push({w[i], k});
        }
    }
    if(cnt != n) return INF;
    return sum;
}

int main()
{
    cin>>n>>m;
    memset(h, -1, sizeof h);
    for(int i = 1; i <= m; i++)
    { 
        int a, b, c;
        cin>>a>>b>>c;
        add(a, b, c);
        add(b, a, c);
    }
    int t = prim();
    if(t == INF) cout<<"impossible"<<endl;
    else cout<<t<<endl;
}
```

### Kruskal

```
C++
#include<iostream>
#include<algorithm>
using namespace std;
const int N = 2e5 + 10;
int p[N], n, m;
struct Edge{
    int a, b, w;
    bool operator < (const Edge& e) const{
        return w < e.w;
    }
};
int find(int x){
    return (p[x] == x ? x : p[x] = find(p[x]));
}
Edge edge[N];
int kul(){
    int cnt = 0, res = 0;
    for(int i = 1; i <= n; i++) p[i] = i;
    for(int i = 1; i <= m; i++){
        int a = edge[i].a, b = edge[i].b, w = edge[i].w;
        if(find(a) != find(b)){
            p[find(a)] = find(b);
            res += w;
            cnt++;
        }
    }
    if(cnt == n - 1) return res;
    else return 0x3f3f3f3f;
}
int main(){
    cin>>n>>m;
    for(int i = 1; i <= m; i++){
        int a, b, w;
        cin>>a>>b>>w;
        edge[i] = {a, b, w};
    }
    sort(edge + 1, edge + m + 1);
    int ans = kul();
    if(ans != 0x3f3f3f3f) cout<<ans<<endl;
    else cout<<"impossible"<<endl;
}
```

## 双端队列

双端队列扩搜 https://www.acwing.com/solution/content/21775/

码下:

```
C++
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <deque>
#include <utility>
using namespace std;
#define x first
#define y second
typedef pair<int, int> PII;
const int N = 600;
int dist[N][N];
char g[N][N];
int n, m;
bool st[N][N];

int bfs()
{
    deque<PII> q;
    memset(dist, 0x3f3f3f3f, sizeof dist);
    memset(st, 0, sizeof st);
    dist[0][0] = 0;
    
    char cs[5] = "\\/\\/";
    int dx[] = {-1, -1, 1, 1}, dy[] = {-1, 1, 1, -1};
    int ix[] = {-1, -1, 0, 0}, iy[] = {-1, 0, 0, -1};
    q.push_back({0, 0});

    while(q.size())
    {
        auto t = q.front();
        q.pop_front();

        int x = t.x, y = t.y;

        if(x == n && y == m) return dist[x][y];

        if(st[x][y]) continue;
        st[x][y] = 1;

        for(int i = 0; i < 4; i++)
        {
            int tx = x + dx[i], ty = y + dy[i];

            if(tx < 0 || tx > n || ty < 0 || ty > m) continue;

            int ga = x + ix[i], gb = y + iy[i];
            int w = (g[ga][gb] != cs[i]);
            int d = dist[x][y] + w;

            if(d <= dist[tx][ty])
            {
                dist[tx][ty] = d;
                if(!w)
                    q.push_front({tx, ty});
                else
                    q.push_back({tx, ty});
            }
        }
    }

    return -1;
}

void solve()
{
    scanf("%d%d", &n, &m);
    for(int i = 0; i < n; i++)
        scanf("%s", g[i]);

    int ans = bfs();
    if(n + m & 1)
        puts("NO SOLUTION");
    else
        printf("%d\n", ans);
}

int main()
{
    int T;
    scanf("%d", &T);
    while(T--)
        solve();
}
```

### 双端队列搜索, 压缩二维坐标, 状态压缩

三维最短路, 双端队列, 多终点 https://www.acwing.com/solution/content/46802/

```
C++
#include <iostream>
#include <cstring>
#include <deque>
#include <set>
#include <utility>
using namespace std;
const int N = 11, M = N * N, E = 400, P = 1 << 10;
typedef pair<int, int> PII;
int n, m, p, K;
int h[M], e[E], w[E], ne[E], idx;
int dist[M][P];
int key[M];
bool st[M][P];
int g[N][N];

set<PII> edges;

void add(int a, int b, int c)
{
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx++;
}

void build()
{
    int dx[] = {-1, 0, 1, 0}, dy[] = {0, 1, 0, -1};

    for(int i = 1; i <= n; i++)
        for(int k = 1; k <= m; k++)
            for(int u = 0; u < 4; u++)
            {
                int x = i + dx[u], y = k + dy[u];
                if(!x || x > n || !y || y > m) continue;
                int a = g[i][k], b = g[x][y];
                if(!edges.count({a, b}))
                    add(a, b, 0), add(b, a, 0);
            }
}

int bfs()
{
    memset(dist, 0x3f3f3f3f, sizeof dist);
    deque<PII> q;
    q.push_back({1, 0});
    dist[1][0] = 0;
    while(q.size())
    {
        auto t = q.front();
        q.pop_front();

        int pos = t.first, state = t.second;

        if (st[pos][state])
            continue;
        st[pos][state] = 1;

        if (pos == n * m)
            return dist[pos][state];

        if (key[pos]) 
        {
            int nstate = state | key[pos];
            if (dist[pos][nstate] > dist[pos][state]) 
            {
                dist[pos][nstate] = dist[pos][state];
                q.push_front({pos, nstate});
            }
        }

        for (int i = h[pos]; ~i; i = ne[i]) 
        {
            int ver = e[i];
            if (w[i] && !(state >> (w[i] - 1) & 1))
                continue;
            if (dist[ver][state] > dist[pos][state]) 
            {
                dist[ver][state] = dist[pos][state] + 1;
                q.push_back({ver, state});
            }
        }
    }
    return -1;
}

int main()
{
    cin>>n>>m>>p>>K;
    for(int i = 1, t = 1; i <= n; i++)
        for(int k = 1; k <= m; k++)
            g[i][k] = t++;
    memset(h, -1, sizeof h);
    while(K--)
    {
        int x1, y1, x2, y2, c;
        cin>>x1>>y1>>x2>>y2>>c;
        int a = g[x1][y1], b = g[x2][y2];
        edges.insert({a, b}), edges.insert({b, a});
        if(c) add(a, b, c), add(b, a, c);
    }

    build();

    cin>>K;
    while(K--)
    {
        int x, y, id;
        cin>>x>>y>>id;
        key[g[x][y]] |= 1 << (id - 1);
    }

    cout<<bfs()<<endl;
}
```