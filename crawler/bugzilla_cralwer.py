#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Created on 2017-07-19 16:04:14
# Project: bugzilla_with_labels_with_priority

from pyspider.libs.base_handler import *

class Handler(BaseHandler):
    crawl_config = {

    }

    @every(minutes=24 * 60)
    def on_start(self):
        cookies = {
            "Bugzilla_login": "33728",
            "Bugzilla_logincookie": "MwzP0WJc3a"
        }

        self.crawl('https://bugzilla.eng.vmware.com/buglist.cgi?query_format=advanced&ctype=&short_desc_type=allwordssubstr&short_desc=&longdesc_type=allwordssubstr&longdesc=&keywords_type=allwords&keywords=&target_milestone_type=allwords&target_milestone=&product=App+Volumes&bug_status=unreviewed&bug_status=new&bug_status=assigned&bug_status=reopened&bug_status=resolved&bug_status=closed', cookies=cookies, callback=self.index_page)

    @config(age=10 * 24 * 60 * 60)
    def index_page(self, response):
        cookies = {
            "Bugzilla_login": "33728",
            "Bugzilla_logincookie": "MwzP0WJc3a"
        }

        for each in response.doc('.w a').items():
            self.crawl(each.attr.href, cookies=cookies, callback=self.detail_page)

    @config(priority=2)
    def detail_page(self, response):
        return {
            "url": response.url,
            "title": response.doc('title').text(),
            "body": response.doc('.bz_comment_content .norm').text(),
            "labels": response.doc('select#product').attr.onchange,
            "priority": response.doc('select#priority option[selected]').text(),
            "bug_severity": response.doc('select#bug_severity option[selected]').text(),
            "cf_type": response.doc('select#cf_type option[selected]').text()
        }
