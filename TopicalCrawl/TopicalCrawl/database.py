#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: actlea
@file: database.py
@time: 16-2-29 下午4:11
"""

import MySQLdb
import sys

def getMysql():
    return MySQLdb.connect(host='localhost', \
                           user='root', passwd='actlea', db='focusedcrawl', port=3306, charset='utf8')

class Basedb:
    db = None
    def connectdb(self):
        try:
            self.db = getMysql()
            print 'connect to the dbserver !'
        except:
            print ":failed connected to db!"
            sys.exit(-1)

    def execsql(self, sql):
        cursor=self.db.cursor()
        try:
            cursor.execute(sql)
            self.db.commit()
        except:
            print '%s is error!' %sql
            self.db.rollback()

    def close(self):
        self.db.close()

    def escapeString(self, s):
        if s is None:
            return 'NULL'
        elif isinstance(s, basestring):
            return '"%s"' % (s.replace('\\','\\\\').replace('"','\\"'))
        else:
            return str(s)

    def insert(self, table_name, dict):
        #build insert sql
        keys = ','.join(dict.keys())
        values = ",".join([self.escapeString(v) for v in dict.values()])
        sql = 'insert into %s (%s) values (%s)' % (table_name,keys,values)
        self.execsql(sql)

    def query(self, sql):
        try:
            cur = self.db.cursor(cursorclass = MySQLdb.cursors.DictCursor)
            cur.execute(sql)
            result = cur.fetchall()
            return result
        except:
            return None

    def drop_table(self, table):
        sql = 'drop table %s' %table
        self.execsql(sql)

    def delete_table(self, table):
        sql = 'delete from %s' %table
        self.execsql(sql)

    def clear_database(self):
        sql = 'drop all tables'
        self.execsql(sql)

    def select(self, table):
        sql = 'select * from %s' %table
        return self.query(sql)

    def set_auto_increment_number(self, table_name, number=1):
        sql = 'alter table %s AUTO_INCREMENT=%d' %(table_name, number)
        self.execsql(sql)


class html_database(Basedb):
    def __init__(self):
        self.connectdb()

    def create_tables(self):
        sql = """
        create table if not exists raw_pages(
         row_id INTEGER PRIMARY KEY auto_increment,
         base_url        varchar(100),
         raw_content     text,
         time            TIMESTAMP
         )ENGINE=MyISAM DEFAULT CHARSET=utf8
        """
        self.execsql(sql)

    def create_seed_url_table(self):
        sql = """
        create table if not exists seed_urls(
         row_id INTEGER PRIMARY KEY auto_increment,
         url varchar(100),
         description text,
         catagory int
         )ENGINE=MyISAM DEFAULT CHARSET=utf8
        """
        self.execsql(sql)

    def inser_raw_page(self, pageItem):
        if pageItem is not None and len(pageItem)>0:
            self.insert('raw_pages',pageItem)
            return True
        return False


if __name__ == '__main__':
    m = getMysql()
    d = html_database()
    d.create_seed_url_table()








