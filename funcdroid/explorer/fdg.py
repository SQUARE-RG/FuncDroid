from hmbot.model.page import Page


class PageNode:
    def __init__(self, index: int, page: Page | None):
        self.type = "page" # "page" 或 "widget"
        self.index = index
        self.page = page
        self.function_description = ""
        self.edges = []
        self.is_visited = False  # 是否已经功能识别
        # self.is_excuted = False  # 是否已经执行过edges


# class FDGNode:
#     def __init__(self, page_node: PageNode, function_description: str = ""):
#         self.function_description = function_description
#         # self.page_nodes = [page_node]

#         self.action_refs: list[tuple[int, int]] = []

#         self.data_in = []  # 当前功能消费的数据
#         self.data_out = []  # 当前功能生产的数据

#         self.data_dependencies = []  # 给哪些节点提供了该节点的数据

#         self.to_test = False  # 是否需要测试

#         self.core_logic = None       
#         # self.test_cases = []   

class FDGNode:

    def __init__(self, function_description: str = ""):
        self.function_description = function_description

        self.action_refs: list[tuple[int, int]] = []

        self.data_in = []
        self.data_out = []

        self.data_dependencies = []
        self.to_test = False

        self.core_logic = None
        
        