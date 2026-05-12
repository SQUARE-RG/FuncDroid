from .vht import VHT, VHTParser
from ..utils.cv import write
import imagehash, cv2
from PIL import Image
import hashlib
from hmbot.utils.cv import encode_image

class Page(object):
    def __init__(self, vht=None, img=None, rsc=None, info=None):
        self.vht = vht
        self.img = img
        self.encoded_img = encode_image(img)
        self.rsc = rsc
        self.info = info
        # 结构哈希值，用于快速比较两个页面的结构相似度
        self.vht_hash, self.feature_set = self._process_vht_recursively(vht._root)
        # 截图哈希值，用于快速比较两个页面的视觉相似度
        self.img_hash = imagehash.phash(Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)))
        self._standardize()
        self.abstract = ""


    def _process_vht_recursively(self, node):
        """
        1. 计算节点的结构哈希值 (自下而上构建)。
        2. 提取节点及其所有子节点的结构化特征集合 (聚合)。
        """
        if not node:
            return ('', set()) # 基本情况：空节点返回空哈希和空集合

        # 1. 提取当前节点自身的属性和特征
        t = node.attribute.get('type', '')
        clickable = node.attribute.get('clickable', '')
        bounds = node.attribute.get('bounds', '')
        
        if isinstance(bounds, list) and len(bounds) == 2 and bounds[0] and bounds[1]:
            w = bounds[1][0] - bounds[0][0]
            h = bounds[1][1] - bounds[0][1]
            bounds_repr = f"{w}x{h}"
        else:
            bounds_repr = "invalid_bounds"
        
        # 当前节点自身的特征
        current_node_feature = f"type={t}|clickable={clickable}|bounds={bounds_repr}"
        # 初始化最终的特征集，包含当前节点的特征
        total_features = {current_node_feature}

        # 2. 递归处理所有子节点
        child_hashes = []
        for child in node._children:
            # 递归调用，获取子节点的哈希和特征集
            child_hash, child_features = self._process_vht_recursively(child)
            
            # 收集子节点的哈希，用于构建当前节点的哈希
            child_hashes.append(child_hash)
            # 聚合子节点的特征集
            total_features.update(child_features)

        # 3. 计算当前节点的哈希值
        child_hashes.sort()
        content_for_hash = f"{t}|{clickable}|{bounds_repr}|{'-'.join(child_hashes)}"
        final_hash = hashlib.md5(content_for_hash.encode('utf-8')).hexdigest()

        # 4. 返回当前节点处理完成的哈希和聚合后的总特征集
        return (final_hash, total_features)

    def _standardize(self):
        if not self.info:
            return
        if self.info.name == '':
            roots = self.vht(bundle=self.info.bundle)
            if len(roots) :
                self.vht = VHT(roots[0])
                self.info.name = self.vht._root.attribute['page']

    def __call__(self, **kwds):
        return self.vht(**kwds)
    
    def _dump(self, id, dir_path):
        vht_file = dir_path + str(id) + '.json'
        img_file = dir_path + str(id) + '.png'
        VHTParser.dump(self.vht, vht_file)
        write(img_file, self.img)
        return (vht_file, img_file)
    
    def _dict(self, vht_file='', img_file=''):
        return {'vht': vht_file,
                'img': img_file,
                'rsc': self.rsc,
                'ability': self.ability,
                'audio_type': self.audio_type,
                'bundle': self.bundle,
                }

    def _is_same(self, page):
        # todo
        if self == page:
            return True
        return False
        if isinstance(new_window, Window):
            vht_sim = self.vht_similarity(new_window)
            img_sim = self.img_similarity(new_window)
            print(f'vht_sim={vht_sim}, img_sim={img_sim}')
        return False

    def vht_similarity(self, page):
        # todo
        vht_sim = 0
        return vht_sim

    def img_similarity(self, page):
        # todo
        img_sim = 0
        return img_sim