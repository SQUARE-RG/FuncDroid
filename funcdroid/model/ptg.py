from hmbot.utils.proto import PageInfo
from .vht import VHT, VHTNode, VHTParser
from .page import Page
from .event import *
import json
import cv2

class PTG(object):
    def __init__(self):
        self.main_pages = []
        self.pages = []
        self._adj_list = {}
        self._visited = {}
    
    def add_main_page(self, page):
        if self.add_page(page):
            self.main_pages.append(page)
            return True
        return False

    def add_page(self, page):
        if self._is_new_page(page):
            self.pages.append(page)
            self._adj_list[page] = {}
            return True
        return False
    
    def add_edge(self, src_page, tgt_page, events):
        self.add_page(src_page)
        self.add_page(tgt_page)
        self._adj_list[src_page][tgt_page] = events
    
    def _is_new_page(self, new_page):
        for page in self.pages:
            if page._is_same(new_page):
                return False
        return True
    
    def _json_list(self, dir_path):
        res = []
        for id in range(len(self.pages)):
            src_page = self.pages[id]
            vht_file, img_file = src_page._dump(id, dir_path)
            edge_list = []
            for (tgt_page, events) in self._adj_list[src_page].items():
                tgt_id = self.pages.index(tgt_page)
                event_list = [event._json() for event in events]
                edge_dict = {'target_id': tgt_id,
                             'events': event_list}
                edge_list.append(edge_dict)
            src_page_dict = src_page._dict(vht_file, img_file)
            src_page_dict['id'] = id
            res.append({'info': src_page_dict, 
                        'edge': edge_list})
        return res

class PTGParser(object):
    @classmethod
    def _extract_node_attributes(cls, node_data):
        # 解析 center 坐标
        center = node_data.get('center', '[0, 0]')
        if isinstance(center, str):
            center = json.loads(center)
        
        # 解析 bounds 坐标
        bounds = node_data.get('bounds', '')
        if isinstance(bounds, str) and bounds:
            # 将 "[0, 0][1200, 2670]" 转换为 [[0, 0], [1200, 2670]]
            import re
            bound_re = r'\[(\d+),\s*(\d+)\]\[(\d+),\s*(\d+)\]'
            match = re.match(bound_re, bounds)
            if match:
                x1, y1, x2, y2 = map(int, match.groups())
                bounds = [[x1, y1], [x2, y2]]
            else:
                bounds = ''
        
        return {
            "bounds": bounds,
            "clickable": node_data.get('clickable', 'false'),
            "longClickable": node_data.get('longClickable', 'false'),
            "selected": node_data.get('selected', 'false'),
            "checkable": node_data.get('checkable', 'false'),
            "checked": node_data.get('checked', 'false'),
            "type": node_data.get('type', ''),
            "id": node_data.get('id', ''),
            "text": node_data.get('text', ''),
            "enabled": node_data.get('enabled', 'true'),
            "focused": node_data.get('focused', 'false'),
            "center": center
        }

    @classmethod
    def parse(cls, device, dir_path):
        with open(dir_path + 'output/ptg.json', 'r') as f:
            json_data = json.load(f)
        ptg = PTG()

        pages = []
        for item in json_data:
            info = item['info']
            vht_path = info['vht']
            img_path = info['img']
            rsc = info['rsc']
            ability = info['ability']
            bundle = info['bundle']
            id = info['id']
            with open(dir_path + vht_path, 'r') as f:
                vht_str = f.read()
            vht_json = json.loads(vht_str)
            vht = VHTParser._parse_hdc_json(vht_json, device)
            img = cv2.imread(dir_path + img_path)
            page_info = PageInfo(bundle=bundle, ability=ability, name=ability)
            page = Page(vht, img, rsc, page_info, id)
            pages.append(page)

        for item in json_data:
            src_id = item['info']['id']
            src_page = pages[src_id]
            for edge in item['edge']:
                tgt_id = edge['target_id']
                tgt_page = pages[tgt_id]
                events = []
                for event in edge['events']:
                    type = event['type']
                    if type == 'Click':
                        node_data = event['node']
                        attrib = cls._extract_node_attributes(node_data)
                        node = VHTNode(device, attrib)
                        event = ClickEvent(node)
                    elif type == 'LongClick':
                        node_data = event['node']
                        attrib = cls._extract_node_attributes(node_data)
                        node = VHTNode(device, attrib)
                        event = LongClickEvent(node)
                    elif type == 'Input':
                        node_data = event['node']
                        attrib = cls._extract_node_attributes(node_data)
                        node = VHTNode(device, attrib)
                        text = event['node']['text']
                        event = InputEvent(node, text)
                    elif type == 'SwipeExt':
                        event = SwipeExtEvent(device, None, event['direction'])
                    elif type == 'Key':
                        event = KeyEvent(device, None, event['key'])
                    elif type == 'StartApp':
                        event = StartAppEvent(device, event['app'])
                    if event:
                        events.append(event)
                ptg.add_edge(src_page, tgt_page, events)
        return ptg


    @classmethod
    def dump(cls, ptg, dir_path, indent=2):
        with open(dir_path + 'ptg.json', 'w') as write_file:
            json.dump(ptg._json_list(dir_path), write_file, indent=indent, ensure_ascii=False)

