#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 16:09:08 2020

@author: zehuali
"""
class ListNode:
     def __init__(self, val=0, next=None):
         self.val = val
         self.next = next
#1
def oddEvenList(head):
        dummy1,dummy2 = ListNode(-1),ListNode(-1)
        odd,even = dummy1,dummy2
        count = 1
        cur = head
        while cur:
            temp = cur.next
            if count%2!=0:
                odd.next = cur
                
                odd = odd.next
            else:
                even.next = cur
                even = even.next
            cur = temp
            count+=1
        even.next = None
        odd.next = dummy2.next
        return dummy1.next
#2
def deleteNodes(self, head: ListNode, m: int, n: int) -> ListNode:
        dummy = ListNode(None)
        dummy.next = head
        i = 0
        while head:
            if i < m-1:
                i += 1
            else:
                j = 0
                while j < n and head.next:
                    head.next = head.next.next
                    j += 1
                i = 0
            head = head.next
        return dummy.next
#3
def deleteNode(self, node):
    node.val = node.next.val
    node.next = node.next.next
#4
def splitListToParts(self, root: ListNode, k: int) -> List[ListNode]:
    res = []
    length = 0
    check = root
    while check:
        length+=1
        check = check.next
    size, remainder = divmod(length, k)
    ans = []
    cur = root
    for i in range(k):
        head = cur
        for j in range(size +(i <remainder)-1):
            cur = cur.next
        if cur:
            temp = cur.next
            cur.next = None
            cur = temp
        ans.append(head)
    return ans
#5
def deleteDuplicates(self, head: ListNode) -> ListNode:
    if not head:
        return head
    check = [head.val,1]
    dummy = ListNode(-1)
    cur = dummy
    head = head.next
    while head:
        if head.val == check[0]:
            check[1]+=1
        else:
            if check[1] == 1:
                cur.next = ListNode(check[0])
                cur = cur.next
            check[0] = head.val
            check[1]=1
        head = head.next
    if check[1] == 1:
        cur.next = ListNode(check[0])
    return dummy.next
#6
def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
    last_tail = None
    new_head = None
    cur = head
    while cur:
        count = 0
        while count<k and cur:
            cur = cur.next
            count+=1
        if count == k:
            new = self.reverse(head,k)
            if new_head == None:
                new_head = new
            if last_tail:
                last_tail.next = new
            last_tail = head
            head = cur
    if last_tail:
        last_tail.next = head
    return new_head
            
def reverse(self,node,k):
    prev = None
    cur = node
    while k:
        temp = cur.next
        cur.next = prev
        prev = cur
        cur = temp
        k-=1
    return prev
#7
def reorderList(self, head: ListNode) -> None:
    slow,fast= head,head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    prev,cur = None,slow
    while cur:
        temp = cur.next
        cur.next = prev
        prev = cur
        cur = temp
    first,second = head,prev
    while second and second.next:
        temp1 = first.next
        first.next = second
        first = temp1
        temp2 = second.next
        second.next = first
        second = temp2
    return head
#8
def nextLargerNodes(self, head: ListNode) -> List[int]:
    cur = head
    stack = []
    res = []
    while cur:
        while stack and stack[-1][1] <cur.val:
            res[stack.pop()[0]-1]=cur.val
        res.append(0)
        stack.append((len(res),cur.val))
        cur = cur.next
    return res
#9
def swapPairs(self, head: ListNode) -> ListNode:
    dummy = ListNode(-1)
    dummy.next = head
    cur = dummy
    while head and head.next:
        slow = head
        fast = head.next
        temp = fast.next
        cur.next = fast
        fast.next = slow
        slow.next = temp
        cur = slow
        head = slow.next
    return dummy.next
#10
#same as q6
#11
def removeElements(self, head, val):
    pre = ListNode(0)
    cur = pre
    head1 = head
    while head1 != None:
        if head1.val != val:
            cur.next = head1
            cur = cur.next
        head1 = head1.next
        if not head1:
            cur.next = head1
    print(cur.val)
    return pre.next
#12
def isValid(self, s):
    a = []
    if len(s)%2 != 0:
        return False
    for i in range(len(s)):
        if s[i] == "(" or s[i] == "[" or s[i] == "{":
            a += s[i]
        else:
            if len(a) == 0:
                return False
            elif s[i] == ")" and a[-1] == "(":
                a.pop()
            elif s[i] == "]" and a[-1] == "[":
                a.pop()
            elif s[i] == "}" and a[-1] =="{":
                a.pop()
            else:
                return False
    if len(a)>0:
        return False
    else:
        return True
#13
def simplifyPath(self, path: str) -> str:
    stack = []
    for element in path.split("/"):
        if element == "..":
            if stack:
                stack.pop()
        else:
            if element != "." and element !="":
                stack.append(element)
    res = ""
    if len(stack) == 0:
        return "/"
    for element in stack:
        res+="/"+element
    return res
#14
def __init__(self):
    self.stack = []
    self.Min = []

def push(self, x: int) -> None:
    self.stack.append(x)
    if len(self.Min) == 0:
        self.Min.append(x)
    else:
        if x <= self.Min[-1]:
            self.Min.append(x)

def pop(self) -> None:
    temp = self.stack.pop()
    if temp == self.Min[-1]:
        self.Min.pop()

def top(self) -> int:
    if len(self.stack) !=0:
        return self.stack[-1]

def getMin(self) -> int:
    if len(self.Min) !=0:
        return self.Min[-1]
#15
def calculate(self, s: str) -> int:
    stack = []
    sign = 1
    total = 0
    curSum = 0
    for i in range(len(s)):
        if s[i].isdigit():
            curSum = curSum*10+int(s[i])
        elif s[i] == "+":
            total += sign*curSum
            sign = 1
            curSum = 0
        elif s[i] == "-":
            total += sign*curSum
            sign = -1
            curSum = 0
        elif s[i] == "(":
            stack.append(total)
            stack.append(sign)
            sign = 1
            total = 0
            curSum = 0
        elif s[i] == ")":
            total+= sign*curSum
            total*= stack.pop()
            total += stack.pop()
            sign = 1
            curSum = 0
    if curSum != 0:
        total += sign*curSum
    return total
#16
def removeDuplicateLetters(self, s: str) -> str:
    stack = []
    last_idx = {c:i for i, c in enumerate(s)}
    seen = set()
    for i, c in enumerate(s):
        if c not in seen:
            while stack and c < stack[-1] and i <last_idx[stack[-1]]:
                seen.remove(stack.pop())
            seen.add(c)
            stack.append(c)
    return "".join(stack)
#17def deserialize(self, s: str) -> NestedInteger:
def nestedInteger():
    num = ''
    while s[-1] in '1234567890-':
        num += s.pop()
    if num:
        return NestedInteger(int(num))
    s.pop()
    lst = NestedInteger()
    while s[-1] != ']':
        lst.add(nestedInteger())
        if s[-1] == ',':
            s.pop()
    s.pop()
    return lst
s = list(' ' + s[::-1])
return nestedInteger()   
#18
def removeKdigits(self, nums: str, k: int) -> str:
    stack = []
    if len(nums) == 0 or k == len(nums):
        return '0'
    for i in range(len(nums)):
        while stack and stack[-1] > nums[i] and k>0:
            stack.pop()
            k-=1
        stack.append(nums[i])
    while stack and k >0:
        stack.pop()
        k-=1
    i = 0
    while i < len(stack) and stack[i] == '0':
        i+=1
    if i == len(stack):
        return '0'
    return "".join(stack[i:])
#19
def removeKdigits(self, nums: str, k: int):
    stack = []
    if len(nums) == 0 or k == len(nums):
        return '0'
    for i in range(len(nums)):
        while stack and stack[-1] > nums[i] and k>0:
            stack.pop()
            k-=1
        stack.append(nums[i])
    while stack and k >0:
        stack.pop()
        k-=1
    i = 0
    while i < len(stack) and stack[i] == '0':
        i+=1
    if i == len(stack):
        return '0'
    return "".join(stack[i:])
#20
def find132pattern(self, nums):
    Min = [0]*len(nums)
    Min[0] = nums[0]
    for i in range(1,len(nums)):
        Min[i] = min(Min[i-1],nums[i])
    stack = []
    for i in range(len(nums)-1,-1,-1):
        if nums[i] > Min[i]:
            while stack and stack[-1] <= Min[i]:
                stack.pop()
            if stack and stack[-1] < nums[i]:
                return True
            stack.append(nums[i])
    return False
#21
def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
    res = []
    stack = []
    dic = {}
    for i in range(len(nums2)):
        while stack and stack[-1] < nums2[i]:
            dic[stack.pop()] = nums2[i]
        stack.append(nums2[i])
    for element in nums1:
        if element in dic:
            res.append(dic[element])
        else:
            res.append(-1)
    return res
#22
def nextGreaterElements(self, nums: List[int]) -> List[int]:
    if len(nums) == 0:
        return nums
    stack = []
    res = [-1]*len(nums)
    for i in range(len(nums)):
        while stack and nums[stack[-1]] < nums[i]:
            res[stack[-1]] = nums[i]
            stack.pop()
        stack.append(i)
    for i in range(len(nums)):
        while stack and nums[stack[-1]] < nums[i]:
            res[stack[-1]] = nums[i]
            stack.pop()
        stack.append(i)
    return res
#23
from collections import Counter
def countOfAtoms(self, formula):
    def parse():
        N = len(formula)
        count = collections.Counter()
        while (self.i < N and formula[self.i] != ')'):
            if (formula[self.i] == '('):
                self.i += 1
                for name, v in parse().items():
                    count[name] += v
            else:
                i_start = self.i
                self.i += 1
                while (self.i < N and formula[self.i].islower()):
                    self.i += 1
                name = formula[i_start: self.i]
                i_start = self.i
                while (self.i < N and formula[self.i].isdigit()):
                    self.i += 1
                count[name] += int(formula[i_start: self.i] or 1)
        self.i += 1
        i_start = self.i
        while (self.i < N and formula[self.i].isdigit()):
            self.i += 1
        if (i_start < self.i):
            multiplicity = int(formula[i_start: self.i])
            for name in count:
                count[name] *= multiplicity

        return count
    self.i = 0
    ans = []
    count = parse()
    for name in sorted(count):
        ans.append(name)
        multiplicity = count[name]
        if multiplicity > 1:
            ans.append(str(multiplicity))
    return "".join(ans)
#24
def asteroidCollision(self, asteroids: List[int]) -> List[int]:
    res = []
    for element in asteroids:
        destroy = False
        while res and element < 0 < res[-1]:
            if abs(element) > abs(res[-1]):
                res.pop()
            elif abs(element) == abs(res[-1]):
                res.pop()
                destroy = True
                break
            else:
                destroy = True
                break
        if destroy == False:
            res.append(element)
    return res
#25
def dailyTemperatures(T):
    res = [0]*len(T)
    q = []
    if len(T)==0:
        return q
    q.append((T[0],0))
    for i in range(1,len(T)):
        while q and T[i] > q[-1][0]:
            temp = q.pop()
            res[temp[1]] = i-temp[1]
            # print(q)
        q.append((T[i],i))
    return res
#26
def backspaceCompare(self, S: str, T: str) -> bool:
    skip_s,skip_t = 0,0
    s_ptr,t_ptr = len(S)-1,len(T)-1
    while s_ptr >=0 or t_ptr >=0:
        while s_ptr >=0 and (S[s_ptr] == "#" or skip_s >0):
            if S[s_ptr] == "#":
                skip_s +=1
            else:
                skip_s-=1
            s_ptr-=1
        while t_ptr >=0 and (T[t_ptr] == "#" or skip_t >0):
            if T[t_ptr] == "#":
                skip_t +=1
            else:
                skip_t-=1
            t_ptr-=1
        if not (S[s_ptr] == T[t_ptr] and s_ptr >=0 and t_ptr >=0):
            return s_ptr == t_ptr == -1
        t_ptr-=1
        s_ptr-=1
    return True
#27
def scoreOfParentheses(self, S: str) -> int:
    stack = []
    cur = 0
    for element in S:
        if element == "(":
            stack.append(cur)
            cur = 0
        else:
            cur = stack.pop() + max(2*cur,1)
    return cur
#28
def decodeAtIndex(self, S: str, K: int) -> str:
    length = 0
    for i in range(len(S)):
        if S[i].isdigit():
            length*= int(S[i])
        else:
            length+=1
    # print(length)
    for ch in reversed(S):
        K%=length
        if K == 0 and ch.isalpha():
            return ch
        if ch.isdigit():
            length//=int(ch)
        else:
            length-=1
#29
def __init__(self):
        self.freq = collections.Counter()
        self.group = collections.defaultdict(list)
        self.maxfreq = 0

def push(self, x):
    f = self.freq[x] + 1
    self.freq[x] = f
    if f > self.maxfreq:
        self.maxfreq = f
    self.group[f].append(x)

def pop(self):
    x = self.group[self.maxfreq].pop()
    self.freq[x] -= 1
    if not self.group[self.maxfreq]:
        self.maxfreq -= 1

    return x
#30
def minAddToMakeValid(self, S):
    ans = bal = 0
    for symbol in S:
        bal += 1 if symbol == '(' else -1
        # It is guaranteed bal >= -1
        if bal == -1:
            ans += 1
            bal += 1
    return ans + bal
#31
#Same as q8
#32
def removeOuterParentheses(self, S):
    res, opened = [], 0
    for c in S:
        if c == '(' and opened > 0: res.append(c)
        if c == ')' and opened > 1: res.append(c)
        opened += 1 if c == '(' else -1
    return "".join(res)