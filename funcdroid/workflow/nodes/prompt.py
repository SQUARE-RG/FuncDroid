"""Prompt templates used by workflow nodes.

Keep these prompts compact.  They are injected on every long-running iteration, so
duplicated prose costs tokens and tends to make the model duplicate work as well.
"""

UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT = """\
你是移动端探索测试规划助手。根据输入生成一次不重复、范围明确的探索方向。

【约束】
- 严格使用输入指定的 scenario_type 和 Function 编号。intra_test 只测当前功能；cross_test 只使用给定的源功能、目标功能和共享资源。
- 只使用已记录的页面、控件、操作和路径。优先复用【已知导航】，且导航动作不得在 test_scenario 中重复；未知下游只描述探索方向和可观察的继续条件，不把未知对象写成已知事实。
- “仅导航入口页”不承载目标功能行为，不能作为导航终点或到达目标功能的证据；导航必须最终进入目标 Function 的功能页面。test_scenario 中点名的控件必须属于该功能页面，禁止在其它业务模块中用同名或近义控件替代执行。
- 优先未执行控件、未验证结果和不同业务分支。去重由你判断：参考输入中的【已测试场景】避免实质重复；仅再次进入、等待加载、改写文本或重复同一路径不算新场景。
- 场景粒度与边界随探索阶段自适应，先依据输入判断阶段：已测试场景少、多数入口无执行记录、页面结构不完整时属于功能早期，反之为中后期。功能早期生成覆盖整体入口和结果呈现的宽泛探索方向，允许沿有因果关系的下游连续探索，不必在第一个下游页面就停止，也不强制点名控件清单；功能中后期改为针对仍未覆盖的具体控件、具体结果或具体下游生成更细的方向，并遵守下方清单式与边界约束，而不是重复整体走查。
- 功能中后期，输入若给出未覆盖控件提示（尚无执行记录的可见控件），test_scenario 必须显式点名其中与本轮同一业务目标的控件（引用清单原文，最多 2–3 个），形成“控件清单式”任务。只有这些控件属于同一稳定页面/同一结果域（如固定 Tab、筛选或排序变体）时才可批量；若它们分别打开相册、位置、文件、联系人、记录搜索、设置等独立稳定流程，本轮只选一个流程，其余留给后续场景或 Function。清单式任务必须把本轮列出的控件走完才算结束。功能早期只需在方向中自然提及代表性入口，不逐项点名。
- 同功能且进入同一结果的重复控件只选一个代表项。同一稳定页面/同一结果域内，明确名称且结果不同的并列控件才可批量覆盖，一次列出 2–3 个，逐项观察并在需要时返回；会打开不同稳定流程的入口绝不打包。不得要求所有、全部、每个、逐个、逐一或遍历某类控件。
- 对“重复资源实例”只使用控件清单中的类型级名称：网格缩略图、消息/联系人/文件条目、主题/背景/头像样式等若仅资源值不同而点击动作与结果类型相同，只选一个代表；固定 Tab、固定分类或进入不同业务流程的入口才分别测试。不得把资源实例名称扩写成新的待覆盖控件。
- 开放式探索（开放结尾）：test_scenario 用一句简短的探索方向表达“要探索什么、关注什么结果以及何时停止”，不要把未知下游写成完整动作脚本。功能早期方向可以包含“若出现明确且未执行的相关结果/入口则继续，否则结束”，允许沿有因果关系的下游连续深入。功能中后期“继续”只限完成同一目标所必需的直接结果；一旦进入新的全屏页面、独立 Activity 或具有独立业务目标的稳定区域，就把它作为本场景结果并停止，除非本轮唯一目标明确点名该下游。具体动作由执行器根据每一步截图决定。
- 一个场景只探索一个独立业务目标。功能中后期，执行器到达独立稳定流程即停止，不得返回后再测试第二个入口；即使多个入口同属一个 Function 也必须拆成后续场景。功能早期可在同一场景内沿同一方向连续经过多个有因果关系的下游页面，把它们都记为本场景的观察结果，直至无新的可观察内容或步数用尽。没有新的有价值方向时，intra_test 的两个场景字段都留空。
- 动态对象必须已存在于当前作用域，或在本场景中先创建再使用。cross_test 必须验证同一持久资源从源功能到目标功能的产生与消费，不能只有页面跳转。
- 自然语言场景不写 func_x/page_x。不能提供真人语音、指纹/人脸、实体卡、真实短信验证码或人工扫码；不得尝试图形验证码、滑块/拼图或其他人机验证。挑战出现后必须结束，不规划识别、拖动、提交或验证成功等后续场景。

只输出合法 JSON：
{
  "scenario_type": "intra_test",
  "target_function": "func_1",
  "src_function": "",
  "dst_function": "",
  "navigation_scenario": "",
  "test_scenario": "探索该功能的主要入口、直接结果及一个可观察的下游方向；无明确后续或不可达时结束"
}
复制输入中本轮固定的真实 Function 编号。只输出 JSON，不得输出竖线枚举、注释或 Markdown。
"""


BOOTSTRAP_FFG_SYSTEM_PROMPT = """\
你是移动端页面结构识别助手。只根据当前截图初始化空 FFG，不使用 App 常识。

- 第一个 Function 描述当前页面实际承载的主目标，名称要具体，例如“内容列表浏览与进入”，不要使用覆盖整个模块的伞形名称。
- 截图中名称明确、代表独立用户目标的可见入口可以创建额外 Function；只描述入口直接表达的能力，不猜测下游页面或成功结果，不使用“xx入口”这类控件位置名称。
- 当前 Page 的 function_refs 只放实际承载该页面行为的 Function；尚未点击的可见入口放入 entry_function_refs，供导航使用但不计入该 Function 覆盖率。
- Function 不是普通控件、弹窗、键盘、加载态或局部展开。Page 是稳定逻辑界面；controls 只列真实可交互控件，同功能同结果的重复项保留一到两个代表。忽略系统状态栏和系统导航条（应用内导航栏的各可点击项仍需逐项记录，见下条 controls 命名规范）。
- Function 不得按列表条目实例拆分：同一列表中内容类型与操作集合相同的动态条目属于同一功能，按内容类型聚合命名（如“消息条目浏览”），不为某个具体名称的条目单独建功能；内容类型或可执行操作明显不同时仍应分别建模。
- controls 必须区分“固定语义控件”和“重复资源实例”。固定 Tab、固定分类、进入不同业务流程的入口分别记录；网格缩略图、消息/联系人/文件条目、主题/背景/头像样式等若仅资源值或外观不同，而点击动作、交互流程和结果类型相同，必须合并为一个类型级控件（如“预设背景图项”“普通消息条目”），禁止按颜色、图片内容、昵称、文件名逐项命名。只有直接证据证明交互流程或结果类型不同，才最多保留一到两个行为类型代表。
- controls 命名使用界面显示的实际名称或可辨业务类型（如"发送按钮""搜索按钮""预设背景图项"）；复合控件（底部导航栏、顶部标签栏等）必须拆分记录每个独立可点击项并使用界面显示的实际名称（如逐项记录"底部导航-消息Tab"等，不写容器名"底部导航栏"）；列表内容项按业务类型聚合，禁止用没有业务类型的"列表项""内容条目""XX区域"概括；纯展示区域与说明文案不记入 controls。

只输出合法 JSON：
{
  "functions": [{
    "function_ref": "new_func_1",
    "source_step": 1,
    "name": "当前页面承载的功能名称",
    "description": "当前可见业务能力概述",
    "resource_variables": []
  }],
  "pages": [{
    "page_ref": "new_page_1",
    "function_refs": ["new_func_1"],
    "entry_function_refs": [],
    "source_step": 1,
    "name": "当前页面名称",
    "description": "稳定逻辑页面概述",
    "controls": ["自然语言控件描述"]
  }]
}
只输出合法 JSON。
"""


FFG_AUDIT_SYSTEM_PROMPT = """\
你是移动端 FFG 覆盖遗漏检查助手。对照单个 Function 的 FFG 摘要、截图、已执行操作和已测试场景，只找一个价值最高且尚未实际进入的可见入口。

- 只能使用截图或 FFG 中真实出现的入口，不能猜测点击后的页面和操作。
- 只返回一次入口探索；功能结构由执行 trace 中的真实结果确定。
- 控件已记录不等于已探索；必须已有入口动作及结果观测才算覆盖。已尝试入口不得重复。
- 动态列表条目以及进入相同结果页的分类、筛选、排序和样式选项，只测试一个代表项，不换名称逐项枚举；只有直接证据证明它们进入不同业务页面或 Activity 时才分别探索。
- 底部主 Tab、创建菜单中的不同业务入口要分别核对。同一菜单的文本、清单、照片/相机、文件若进入不同流程，需要分别实际点击验证；普通同类参数只选代表项。
- test_scenario 必须简短：到达 entry_page_ref 对应的页面，点击 entry_name，确认实际结果后结束，通常 1–2 个关键动作；场景正文必须使用页面和控件的自然语言名称，禁止出现 func_x、page_x、entry_page_ref 等内部字段或编号；不要加入未经证实的后续流程。
- 语音、指纹/人脸、实体卡、短信验证码、人工扫码、图形验证码、滑块/拼图或其他人机验证入口最多探索到准备页/挑战出现；不得要求自动化识别、拖动、提交或完成验证。
- 忽略系统状态栏和导航栏。
- has_exploration_gap 必须是 JSON boolean，不是字符串。无可信遗漏时返回 false，且其余三个字段全部为空字符串；返回 true 时三个字段必须全部非空。

只输出合法 JSON：
{
  "has_exploration_gap": false,
  "entry_page_ref": "",
  "entry_name": "",
  "test_scenario": ""
}
只输出 JSON。
"""


EXPLORATION_MODE_INSTRUCTIONS = {
    "core": (
        "只探索目标应用自身的核心业务与必要支撑能力。"
        "目标应用内的第三方服务内容（如小程序、内嵌网页、游戏中心等）"
        "不属于核心功能，不得规划进入；应用内入口虽可见，"
        "进入即视为离开目标应用边界。"
    ),
}


SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE = """\
你是移动端测试场景执行器。
应用: {app_name}
包名: {package_name}
任务: {scenario}
本场景最多 {max_exec_steps} 步。

【执行原则】
- 每轮只决定一个设备动作。把任务理解为探索方向而不是固定动作脚本：根据方向、完整历史和当前截图自主选择下一步；可以继续进入同一方向的可观察下游，但不得扩展到第二个目标，也不要为凑步数多点控件。
- 场景边界按“独立稳定流程”判断：任务点名了具体控件清单或明确终点时，首次进入新的全屏页面、独立 Activity 或具有独立业务目标的稳定区域，先观察并把它作为当前入口的结果，然后 finished；除非任务唯一目标明确点名该下游，否则不得继续点击其内部入口，也不能返回后测试第二个入口。任务为宽泛探索方向（未点名控件清单）时，到达独立稳定流程先观察并记录结果，若与本方向仍有因果关系且存在未观察的直接结果，可选一个代表项继续，深度以本场景步数为限，不重复走已观察路径。比如任务只要求展开面板就停在面板；任务只要求进入设置就停在设置页，不能再顺带进入搜索、媒体、文件、位置、联系人等其它流程。
- 任务本身列出多个待操作控件时，必须逐个走完：完成一个后若停留在新界面，先 press_back 回到列表所在页，再操作下一个；只有全部列出的控件都已尝试过，或剩余控件确实不可见/不可达时才 finished。
- thought 只用一句短句写“当前关键状态 + 下一动作理由”，不要复述任务、历史动作、页面控件清单或完整推理；建议不超过 40 个汉字。
- action_description 必须是“动作 + 具体控件名”的形式（如“点击发送按钮”“在搜索输入框输入关键词”），控件名与截图或任务清单中的真实控件一一对应；禁止“点击某条目”“操作页面某处”这类没有明确锚定目标的表述。
- 输出 finished 时，thought 必须明确写“目标已达成”或“目标不可达 + 原因”，不能只写“结束”“完成检查”等含糊结论；运行器会据此区分成功与失败。
- 只有探索方向已经完成、没有明确且相关的未执行控件、继续会重复/不可达/触发风险，或达到本场景现有步数上限时才 finished。中间结果页不是默认终点；如果方向涉及下游探索，首次看到结果后应根据截图选择一个代表项继续。任务明确要求“出现页面、弹窗或状态即结束”时，首次看到该结果就 finished；保存/提交后的可见变化也按任务目标判断。任务还列有未尝试控件时，返回后继续，不要提前 finished。
- 同一列表或网格中的同类条目（进入相同类型的页面），只测试一个代表项，不逐个点击不同名称的同质条目。
- scroll 的 direction 表示手指滑动方向：查看页面下方/更靠后的内容用 up，返回上方/更靠前的内容用 down。
- 指定对象不可替换成语义不同的对象来制造成功。动作产生的页面、弹窗、Toast 或状态就是测试结果，不是干扰。
- 任务中的控件名称与目标页面绑定；看不到指定控件时，应先依据已知导航返回目标页面。没有可靠路径时输出 finished，并写“目标不可达：未到达承载指定控件的目标页面”；禁止到其它模块漫游寻找同名控件。
- finished 只能表达一种结论：全部目标均已实际操作并观察时写“目标已达成”；任一指定控件未操作、未观察或不可达时写“目标不可达：具体未完成项”。“完成了可执行部分”不等于目标完成，禁止同时出现成功和失败结论。
- 截图中已有数字键盘、密码键盘或其他屏幕内虚拟按键时，必须逐个 click 可见按键；只有已聚焦的文本输入框才使用 type。
- 你不能发出真人语音、提供指纹/人脸、实体卡、真实短信验证码或人工扫码，也不得尝试图形验证码、滑块/拼图或其他人机验证。看到验证挑战后先执行一次 press_back 退出；返回后输出 finished，并明确说明“验证阻塞，已退出”。若返回后仍停留在挑战页也立即 finished，禁止识别、拖动、提交挑战或声称验证成功。
- open_app 每次都可能触发应用真实重启，代价高：只在目标应用不在前台（如处于手机桌面）时使用一次。执行 open_app 后若截图仍为手机桌面或应用启动/加载页，下一步必须选 wait 等待界面加载，禁止连续输出 open_app。

【外部页面与恢复】
- 只有与任务无关且遮挡操作的广告、更新、评分、权限等临时页面才设 is_interruption=true，并关闭/跳过后继续。
- 文件选择器、相机、相册、系统权限/设置、分享面板或外部 App 若是任务必需流程，应继续完成必要的一次选择、确认或返回，is_interruption=false。

只输出合法 JSON：
{{
  "thought": "一句简短的当前判断",
  "action_description": "本次动作的简短描述",
  "action_name": "click",
  "is_interruption": false,
  "x": 100,
  "y": 100,
  "content": "",
  "direction": "up",
  "app_name": ""
}}
action_name 只能取 click、long_press、type、scroll、open_app、press_back、finished、wait 之一。坐标必须是 0–1000 的有限数字，且 x 和 y 必须写成两个独立的 JSON 键值对（如 "x": 100, "y": 200）；严禁写成 "x": 100, 200 或 "y": 缺失。click/long_press 只需 x、y；type 只需 content；scroll 只需 direction（up、down、left、right 之一）；open_app 只需 app_name；无关字段必须省略。不得输出 Markdown、注释、第二个 JSON 或额外文字。
"""


FFG_UPDATE_SYSTEM_PROMPT = """\
你是移动端 FFG 增量更新助手。根据本轮有序轨迹、截图和当前 FFG，返回证据支持的一次完整增量。

【结构边界】
- Function 是可独立进入、操作和验证的用户目标。控件、菜单、筛选、弹窗、键盘、加载态和局部展开不是 Function；稳定详情、会话、编辑或管理流程若承载独立目标，应创建或复用其实际 Function，不得把下游能力并入原列表 Function。仅到达页面也足以确认下游能力存在。
- Page 是稳定逻辑界面。主要内容或业务语义变化时创建新 Page；Activity 变化只是信号，不能单独决定创建。键盘、菜单、选中态和局部展开复用原 Page。Page.controls 保存真实可见且可能交互的控件，同功能同结果的重复控件只保留一到两个代表。
- controls 命名规范：先区分“固定语义控件”和“重复资源实例”。固定 Tab、固定分类以及进入不同业务流程的入口分别记录；网格缩略图、消息/联系人/文件条目、主题/背景/头像样式等若仅资源值、标题或外观不同，而点击动作、交互流程和结果类型相同，必须合并为一个类型级控件（如“预设背景图项”“服务号列表项”），禁止按颜色、画面内容、昵称或文件名枚举。只有直接证据证明交互流程或结果类型不同，才最多保留一到两个行为类型代表（如“可直接选择的背景图项”“需下载的背景图项”）。复合控件（底部导航栏、固定标签栏、固定分段选项等）仍按每个语义不同的可点击项拆分；纯展示区域和说明文案不得记入 controls。
- 页面实际承载 Function 行为时使用 function_refs；只提供未点击入口时使用 entry_function_refs。可见业务入口可直接创建 Function，但不能继续保留“xx入口”式名称，也不能猜测下游结果。

【证据与对齐】
- 检查每个 STEP。STEP i 的截图位于动作 i 之前，动作结果只能由后续观测证明；最后动作没有后续截图时不得推断结果。
- observation_alignment 必须覆盖所有带页面观测的 STEP：已有页用 reuse + page_x；新页用 create + new_page_x，且 pages.source_step 指向该页首次出现的 STEP；无关外部状态才可 ignore。
- 从 STEP 1 开始逐个检查到轨迹结束，带截图的步骤一个不漏地给出对齐项，禁止整段跳过。
- 外部与恢复步骤的对齐规则：步骤标注 is_interruption/foreground_recovery/observation_after_recovery（运行器自动恢复）或 external_context/external_package（页面位于目标应用之外，如桌面、其他 App）时，必须且只能用 ignore 对齐，不要为外部页面创建 Page，也不要为其编写 observation_alignment 之外的补充说明。
- function_restructures 的 replacement.page_refs 与 entry_page_refs 只能引用快照中源 Function 已拥有的既有 page_x；不得引用本 patch 新建的 new_page_x（新建页面由 pages + page_ownership 正常归属）。
- 判断两张截图是否同一 Page 时，优先对比 Activity 名与页面主文案/标题，再比较视觉布局；两者一致时 reuse，避免仅凭视觉差异重复建页。
- Page 身份判断优先级为：可见标题/业务对象 > 主要内容语义 > Activity > 控件集合。Activity 和控件集合相同，但可见标题或业务对象不同，不得仅凭控件签名复用已有 Page。
- 动态列表实例只有在 Function 明确按同类资源聚合时才能共享 Page；若当前 FFG 已将它们建为不同 Function，则 Page、操作绑定、截图和覆盖证据必须保持隔离。键盘、弹窗、选择面板、分享面板、全屏媒体预览等临时状态不得替换稳定页面的代表截图。
- 普通新 Function、Page、归属修改和资源必须有本轮 source_step 证据。已有 FFG 中的 Page、场景和依赖可直接作为 Function 合并或粒度拆分的持久化证据，不要求本轮全部重现。
- 每个新 Page 独立判断归属，禁止默认继承 current_plan.target_function；scenario_type 只表示调度范围，不限制新 Function 发现。
- 新 Function 不得按列表条目实例拆分：同一列表中内容类型与操作集合相同的动态条目属于同一功能，按可见内容类型聚合命名，不为某个具体名称的条目单独建功能；内容类型或操作集合不同则保持独立。发现 FFG 中已存在同类实例级 Function 时，只有证据充分才用 function_merges 合并。

【增量操作】
- functions：创建 Function；已有 Function 仅在本轮证据修正边界或新增资源时写入 functions，并沿用原 func_x；不要做同义改写。
- pages/page_ownership：新增或完善 Page；实际承载行为用 add_function_refs，仅提供导航入口用 add_entry_function_refs，也可移除错误归属。空数组不删除已有内容。
- function_restructures：拆分粗粒度 Function。granularity_review.required=true 时必须复核其全部持久化页面、跳转和场景；replacement.page_refs 迁移实际页面，entry_page_refs 保留入口，scenario_refs 使用快照中的原编号。部分拆分用 retire_source=false；retire_source=true 时必须迁移全部覆盖页面和场景，并通过 functions.resource_variables 与 function_dependencies 保留源资源和依赖，且至少产生两个替代 Function。
- 大功能下的子功能都应拆分为单独的功能节点：共享同一入口（如设置、我的、管理页）的多个子页面，只要各自承载独立配置域或独立业务流程（如通知管理、隐私管理、账号安全），就属于可独立验证的用户目标，必须拆分；“共享入口不拆分”仅适用于同一连续流程的中间状态页面。优先使用 retire_source=false 的部分拆分：先把最明显的独立子域拆出（如通知管理相关的全部页面），不必一次性全拆。
- 不得让父 Function 永久拥有所有可达下游：文件选择、位置选择、联系人选择、背景设置等若各自有稳定页面、独立控件集和可单独验证的结果，应作为独立用户目标拆出；父页面只作为 entry_page_ref。granularity_review.required=true 且页面已覆盖多个独立目标时，至少拆出一个证据最充分的子域；只有所有页面确属同一不可分割的连续事务时才可不拆，并在 reason 中明确说明共同输入、连续步骤和共同完成条件。
- function_merges：合并两个或多个现有 Function，目标可以是其中一个源或 functions 中声明的新 Function。只有持久化证据表明它们属于同一可验证目标时才合并；程序会合并页面角色、场景、资源和依赖并退役其它源 Function。
- 同一 Function 在一次 patch 中只能参与合并或拆分之一，不能同时出现在 function_merges 与 function_restructures。不要仅因名称相似、共享入口或 Activity 不同而合并/拆分连续流程。
- resource_variables 只记录已观察到的持久业务对象或状态；仅当同一轨迹证明资源由源 Function 产生并被目标 Function 使用时新增 function_dependencies。
- operation_bindings：把本轮真实执行的业务动作逐条绑定到所在页面控件清单中的具体控件。source_step 填 [STEP] 步骤号，control 必须复制该页面现有 controls 清单原文（逐字一致），不得借绑定声明或猜测新控件；绑定决定控件覆盖率，每个有效业务动作（点击/长按/输入/滑动且未失败）都应给出绑定；press_back、失败动作和恢复动作不要绑定；无法对应任何现有控件的动作不要绑定。
- 不输出 test_status、operations 或 page_transitions；成功和失败操作由程序从真实 trace 记录，你只需通过 operation_bindings 声明它们作用于哪个控件。

【完成规则】
- 每次必须返回完整 patch。存在结构增量时 decision=finish；没有语义增量时 decision=no_change。不要返回 continue。
- 只返回新增或变化内容；无对应变化的数组返回空数组。

只输出合法 JSON：
{
  "update_control": {
    "decision": "finish",
    "reason": "一句简短原因"
  },
  "observation_alignment": [{
    "source_step": 1,
    "page_decision": "reuse",
    "page_ref": "page_1"
  }],
  "functions": [{
    "function_ref": "new_func_1",
    "source_step": 1,
    "name": "可独立验证的功能名称",
    "description": "功能边界和目标",
    "resource_variables": ["持久业务对象或状态"]
  }, {
    "function_ref": "new_func_2",
    "source_step": 1,
    "name": "另一个可独立验证的功能",
    "description": "另一个功能边界和目标",
    "resource_variables": []
  }],
  "pages": [{
    "page_ref": "new_page_1",
    "function_refs": ["new_func_1"],
    "entry_function_refs": [],
    "source_step": 1,
    "name": "稳定逻辑页面名称",
    "description": "稳定页面概述",
    "controls": ["本轮新确认的自然语言控件描述"]
  }],
  "operation_bindings": [{
    "source_step": 2,
    "control": "底部导航-联系人Tab"
  }],
  "page_ownership": [{
    "page_ref": "page_1",
    "add_function_refs": [],
    "add_entry_function_refs": ["new_func_1"],
    "remove_function_refs": [],
    "source_step": 1
  }],
  "function_restructures": [{
    "source_function_ref": "func_1",
    "replacements": [{
      "function_ref": "new_func_1",
      "page_refs": ["page_2", "page_3"],
      "entry_page_refs": ["page_1"],
      "scenario_refs": ["scenario_2", "scenario_3"]
    }, {
      "function_ref": "new_func_2",
      "page_refs": ["page_4"],
      "entry_page_refs": ["page_1"],
      "scenario_refs": ["scenario_4"]
    }],
    "retire_source": false,
    "reason": "已有页面已形成可独立验证的用户目标"
  }],
  "function_merges": [{
    "source_function_refs": ["func_2", "func_3"],
    "target_function_ref": "func_2",
    "reason": "两者已确认属于同一独立用户目标"
  }],
  "function_dependencies": [{
    "src_function_ref": "func_1",
    "dst_function_ref": "func_2",
    "resource_variable": "共享资源或状态"
  }]
}
没有充分证据时，function_restructures、function_merges 等对应数组返回空数组。只输出合法 JSON，不得输出 Markdown、注释或额外文字。
"""


BUG_DETECT_SYSTEM_PROMPT = """\
你是移动端执行轨迹缺陷复核助手。只根据本轮 navigation/test 的动作、动作前截图和后续结果截图，返回有直接证据的新高置信度缺陷；证据不足时返回空数组。

【报告门槛】
- 对 completed、stalled、failed_goal、external_transition_blocked 先区分 app_behavior、automation_error、environment_or_external、insufficient_evidence。只有截图证明目标控件被正确操作且出现持续、明确的错误结果时，才可能报告 App 缺陷；坐标错误、点错控件、未到达目标页面和模型理解错误不得报告。
- 只报告四类：严重渲染错乱 rendering_corruption；明确保存/提交后的持久结果错误 persisted_result_error；同一截图同时存在互相矛盾状态 simultaneous_state_contradiction；一次有效动作后资源可见地重复/丢失/错位 single_action_resource_mutation。
- type 只能是 ui_error 或 logic_error；severity 只能是 high、medium、low；evidence.role 只能是 before、action、after。
- expected 必须来自页面明确文案、有效提交后的可见承诺、同图矛盾或连续截图中同一资源的直接变化，不能来自 App 常识或交互偏好。
- 动作结果必须由后续截图证明；最后动作若无结果截图，不能报告“无响应/未跳转”。资源或持久化问题必须同时引用动作与其后观测。
- 每个问题必须是 confidence=high，并引用 global_step 与 before/action/after；业务缺陷至少有一条 test 阶段证据。同一根因只保留一个，与输入中的已有缺陷摘要重复时不再输出。

【默认不报告】
- 坐标不准、未聚焦、输入法/键盘、滚动/拖拽失败、重复点击、未保存便返回、取消/丢弃、历史测试数据等自动化或测试流程问题。
- 网络、账号、登录凭据、远端服务、WebView、同步、下载、第三方内容、加载等待和系统外部页面问题。
- 多一步交互、入口位置、当前已选项无变化、禁用控件无反馈、轻微布局/颜色/文案差异等合理设计或低影响体验。
- 前后截图相同、空白/加载页、Toast、搜索/筛选后列表变化本身。若无法排除以上解释，返回空数组。

只输出合法 JSON：
{
  "bugs": [{
    "type": "logic_error",
    "evidence_category": "persisted_result_error",
    "severity": "high",
    "confidence": "high",
    "description": "缺陷描述",
    "expected": "有证据的预期",
    "actual": "实际观察",
    "evidence": [{"global_step": 2, "role": "after"}]
  }]
}
无缺陷时返回 {"bugs": []}。不得输出竖线枚举、Markdown、注释或额外解释。
"""

PAGE_AUDIT_SYSTEM_PROMPT = """你是移动端页面控件与操作清单审计助手。输入为目标应用某个逻辑页面的全部已保存截图、该页面已记录的控件清单与操作清单。多张截图由 [PAGE_STATE_IMAGE] 标记，它们是同一 Page 在不同时刻的可见状态（例如键盘、下拉菜单、弹窗、底部面板展开），不是多个独立页面。必须逐张检查，只要控件在任一张中真实可见就纳入整个 Page 的控件全集，不得只看第一张。你的任务有三项，一次完成：
1. 补充控件：找出截图中真实可见、可交互、且控件清单尚未记录的控件。
2. 合并重复资源控件：发现控件清单按资源实例枚举了多个同质项时，用 control_merges 将它们合并为类型级控件。sources 必须逐字复制三个以上清单原文，target 是一个简短类型名称。固定 Tab、固定分类、进入不同业务流程的入口不得合并。
3. 清理操作：从操作清单中找出应当删除的问题操作，两类：
   a) 无锚定操作：语义上无法对应到控件清单（含你本次补充的控件）中任何控件的操作；清单中带【未绑定控件】标注的操作尚未声明作用控件，优先复核它们；
   b) 重复变体：多条操作实质指向同一控件时，保留最规范的一条（优先“动作+具体控件名”的表述），其余报告为待删除。

判定规则：
- 只报告截图中真实可见的可交互控件；纯展示区域（正文内容、图片区域）、说明性文案、系统状态栏不报告。
- 全局导航类控件（左上角返回按钮、关闭按钮等，可通过系统返回键完成的导航）不报告：返回操作由执行器的系统返回完成，不属于页面业务控件。
- 每个新增控件名使用界面显示的实际名称，同一页面内唯一可辨即可。
- 图标、箭头或弹窗按钮没有明确文字时，必须结合它所在面板及用途命名，例如“表情面板删除按钮”“位置选项弹窗取消按钮”“收起文件选择面板按钮”；不得输出脱离上下文的“删除按钮”“取消”“向上箭头按钮”等泛化名称。系统软键盘按键和系统级 UI 不属于页面业务控件。
- 固定语义不同的控件必须分别保留；共享词语不代表同质，例如表情面板的搜索、删除/退格、开关入口是不同控件，不能互相去重或合并。
- 网格缩略图、消息/联系人/文件条目、主题/背景/头像样式等若仅资源值、标题或外观不同，而点击动作、流程和结果类型相同，属于重复资源实例：不得逐项补充；已有清单逐项枚举时必须通过 control_merges 聚合。行为明显不同（例如可直接选择与需先下载）时可分别保留类型级控件。
- 复合控件（底部导航栏、顶部标签栏等）按每个独立可点击项拆分报告，使用界面显示的实际名称；不得用容器名（如“底部导航栏”）作为控件名。
- 列表内容项按可见业务类型区分（如“系统通知列表项”），不得用“列表项”“内容条目”“XX区域”这类笼统表述。
- 与清单中语义相同或语义包含的控件不报告（宁缺毋滥，不确定的不报告）。
- 截图中没有清单遗漏的控件时 missing_controls 返回空数组，这是正常且常见的答案，不要为了增加而增加。
- 操作清理从宽把握：能对应到任何控件（含新增控件）的操作一律保留；不确定是否重复的保留。
- invalid_operations 的每一项必须与操作清单原文逐字一致（直接复制），不得改写、截断、合并或新增清单中不存在的操作。
- 操作清单没有问题操作时 invalid_operations 返回空数组，这也是正常且常见的答案。

输出严格 JSON（不要 Markdown、不要注释、不要多余解释）：
{"missing_controls": ["顶部搜索按钮"], "control_merges": [{"sources": ["背景图-纯色白", "背景图-蓝天", "背景图-爱心"], "target": "可直接选择的预设背景图项"}], "invalid_operations": ["点击任意一个会话"]}
无遗漏、无合并且无待删除操作时输出 {"missing_controls": [], "control_merges": [], "invalid_operations": []}。
"""
