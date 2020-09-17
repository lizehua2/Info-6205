#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 22:32:55 2020

@author: zehuali
"""
#1
class ListNode:
     def __init__(self, val=0, next=None):
         self.val = val
         self.next = next
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummy = ListNode(-1)
        dummy.next = head
        #cur =dummy
        slow = dummy
        fast = dummy
        while n:
            fast = fast.next
            n-=1
        while fast.next:
            slow = slow.next
            fast = fast.next
        # print(slow)
        slow.next = slow.next.next
        return dummy.next
#2
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        dummy = ListNode(-float('inf'))
        dummy.next = head
        cur = dummy
        prev = None
        while cur != None:
            if prev == None or cur.val != prev.val:
                prev = cur
                cur = cur.next
            else:
                cur = cur.next
                prev.next = cur
        return dummy.next
#3
    def partition(self, head: ListNode, x: int) -> ListNode:
        dummy1,dummy2 = ListNode(float('inf')),ListNode(float('inf'))
        head1,head2 = dummy1,dummy2
        cur = head
        while cur:
            if cur.val < x:
                head1.next = cur
                head1 = head1.next
            else:
                head2.next = cur
                head2 = head2.next
            temp = cur.next
            cur.next = None
            cur = temp
        head1.next = dummy2.next
        return dummy1.next
#4
    def insert(self, head: 'Node', insertVal: int) -> 'Node':
        new_node = ListNode(insertVal)
        # print(new_node.next.val)
        if not head:
            new_node.next = new_node
            return new_node
         
        cur = head
        while True:
            if cur.val > cur.next.val and (insertVal >= cur.val or insertVal <= cur.next.val):
                break
            elif cur.val <= insertVal <= cur.next.val:
                break
            elif head == cur.next:
                break
            cur = cur.next
        new_node.next = cur.next
        cur.next = new_node
        return head
#5
    def getDecimalValue(self, head: ListNode) -> int:
        res = 0
        while head:
            res = res*2+head.val
            head = head.next
        return res