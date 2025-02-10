target_txt = [
    """似芙蓉涉水而来，旖旎独特；如野菊淡然绽放，暗香盈袖。汉字，你在中国这片原野上落地、生根，从此成了中国心，成了中国魂，成了我心中那段不了的汉字情。汉字，于春之暮野，我们邂逅。你向我伸出手，眼波流转，微笑蔓延。从此，我沉醉。

汉字，犹记得初感受你指尖的温存，被你牵引着走向未知的远方，身后是流光飞舞的红尘岁月。在那些遥远的往事里，是你留下的快乐悲伤，惊奇瑰丽。你的故事，从春说到秋，从绿说到黄，编织着神奇的童话，承载着对世界的最初的感知。
汉字，童年是你在短笛里谱写的牧歌，单纯而恬静。那些小水洼，那些烂泥巴，那些在雨天光着的小脚丫，因为有你陪伴，它们便都开成我心口的白色小花，芬芳那一颗赤诚的中国心。
汉字，当我马不停蹄地向前，终于跑进成长的劫，触摸到那明媚而忧伤的青春。此时，是你，是你在我迷茫时用那闪亮的字句点燃那已微弱的心火；是你在我莫名落泪时用温柔的歌抚平我心的是疼痛；也是你教会我，在年轻的心壁上默默雕琢一种激情，一种信仰，一种向上的力量；更是你，告诉我，既然活着就要像水，点点滴滴都是真实的生命。
汉字，青春是你调制的一杯鸡尾酒，眩目而浓烈。喝下那杯酒，我丢掉那些浮躁的喧嚣与空虚，和你一起或喜，或悲，或怒，或痴，任我血脉里那澎湃的中国情燃成熊熊烈火。汉字，当我满了黑发，长了腰肢，是你牵紧我的手，向那庄严的历史与人生致敬。你踏着岁月的尘埃，碾碎时间的凹凸走进历史的风雨中，我，亦无悔相随。痴迷了，沉醉了，我沦陷在你的天地里。汉字，你温柔的发梢吹来历史的夜风，讲述一个个千年古老的故事；你多情的双眸噙满泪水，赞叹着一次次文明的奇迹；你滑润的肌肤令我酥麻入心，感受着那一段段惊心动魄的传奇。五千年的灿烂辉煌，在你心口浅吟低唱，余晖后的屈辱也融进你掌心的纹路，黯然神伤。微醺的""",
    # =====================
    """import Levenshtein

# 目标文本
A = "目标文本"
# 模型1输出的文本
B = "模型1输出的文本"
# 模型2输出的文本
C = "模型2输出的文本"

# 计算编辑距离
distance_B = Levenshtein.distance(A, B)
distance_C = Levenshtein.distance(A, C)

# 输出结果
print(f"编辑距离（A与B）: {distance_B}")
print(f"编辑距离（A与C）: {distance_C}")

# 判断哪个模型更相似
if distance_B < distance_C:
    print("模型1的输出文本与目标文本更相似。")
elif distance_B > distance_C:
    print("模型2的输出文本与目标文本更相似。")
else:
    print("两个模型的输出文本与目标文本同样相似。")""",
    # =====================
    """Overview
Our English language classes are always full of fun!

ESF Language uses everything from stories and role-plays to art and debate. Our English course develops students' language skills through a wide range of activities. This may include projects, speeches, and games. Hence, this instils a love of learning.

The topic-based English programmes bring books alive! Students have the opportunity to act out their favourite stories. Additionally, they may also be involved in games related to the stories. By that, this captures the imagination of our students. Plus, they will develop the four main language skills: reading, writing, speaking and listening.

What's more, we encourage students to express themselves in English freely. Through familiar contexts, they will be communicating entirely in English. Meaning, whatever it is they do, they must do so in English. Additionally, they get to do interesting new activities, be part of discussions, and more! On top of this, ESF offers an energetic & dynamic learning programme. We discourage sitting still on their textbooks. Instead, we incorporate games and crafts with a traditional English language course. Textbooks and a workbook only add to our programmes.


For specific skills development, we have Writing and Phonics programmes which offer more specialised lessons.""",
]
transfomrer_txt = [
    """似芙蓉涉水而来，曳曳独特；如野菊淡然绽放，暗香盈袖。汉字，你在中国这片原野上落地、生根，从此成了中国心，成了中国魂，成了我心中那段不了的汉字情。汉字，于春之暮野，我们邂逅。你向我伸出手，眼波流转，微笑蔓延。从此，我沉醉。

汉字，犹记得初感受你指尖的温存，被你牵引着走向未知的远方，身后是流光飞舞的红尘岁月。在那些遥远的往事里，是你留下的快乐悲伤，惊奇瑰丽。你的故事，从春说到秋，从绿说到黄，编织着神奇的童话，承载着对世界的最初的感知。

汉字，童年是你在短笛里谱写的牧歌，单纯而恬静。那些小水洼，那些烂泥巴，那些在雨天光着的小脚丫，因为有你陪伴，它们便都开成我心口的白色小花，芬芳那一颗赤诚的中国心。

汉字，当我马不停蹄地向前，终于跑进成长的劫，触摸到那明媚而忧伤的青春。此时，是你，是你在我迷茫时用那闪亮的字句点燃那已微弱的心火；是你在我莫名落泪时用温柔的歌抚平我心的是疼痛；也是你教会我，在年轻的心壁上默默雕琢一种激情，一种信仰，一种向上的力量；更是你，告诉我，既然活着就要像水，点点滴滴都是真实的生命。

汉字，青春是你调制的一杯鸡尾酒，眩目而浓烈。喝下那杯酒，我丢掉那些浮躁的喧嚣与空虚，和你一起或喜，或悲，或怒，或痴，任我血脉里那澎湃的中国情燃成熊熊烈火。汉字，当我满了黑发，长了腰肢，是你牵紧我的手，向那庄严的历史与人生致敬。你踏着岁月的尘埃，碾碎时间的凹凸走进历史的风雨中，我，亦无悔相随。痴迷了，沉醉了，我沦陷在你的天地里。汉字，你温柔的发梢吹来历史的夜风，讲述一个个千年古老的故事；你多情的双眸噙满泪水，赞叹着一次次文明的奇迹；你滑润的肌肤令我酥麻入心，感受着那一段段惊心动魄的传奇。五千年灿烂辉煌，在你心口浅吟低唱，余晖后的屈辱也融进你掌心的纹路，黯然神伤。微醺的""",
    # =====================
    """import Levenshtein

# 目标文本
A = "目标文本"

# 模型1输出的文本
B = "模型1输出的文本"

# 模型2输出的文本
C = "模型2输出的文本"

# 计算编辑距离
distance_B = Levenshtein.distance(A, B)
distance_C = Levenshtein.distance(A, C)

# 输出结果
print(f"编辑距离（A与B）：{distance_B}")
print(f"编辑距离（A与C）：{distance_C}")

# 判断哪个模型更相似
if distance_B < distance_C:
    print("模型1的输出文本与目标文本更相似。")
elif distance_B > distance_C:
    print("模型2的输出文本与目标文本更相似。")
else:
    print("两个模型的输出文本与目标文本同样相似。")""",
    # =====================
    """
Overview

Our English language classes are always full of fun!

ESF Language uses everything from stories and role-plays to art and debate. Our English course develops students' language skills through a wide range of activities. This may include projects, speeches, and games. Hence, this instils a love of learning.

The topic-based English programmes bring books alive! Students have the opportunity to act out their favourite stories. Additionally, they may also be involved in games related to the stories. By that, this captures the imagination of our students. Plus, they will develop the four main language skills: reading, writing, speaking and listening.

What's more, we encourage students to express themselves in English freely. Through familiar contexts, they will be communicating entirely in English. Meaning, whatever it is they do, they must do so in English.

Additionally, they get to do interesting new activities, be part of discussions, and more! On top of this, ESF offers an energetic & dynamic learning programme. We discourage sitting still on their textbooks. Instead, we incorporate games and crafts with a traditional English language course. Textbooks and a workbook only add to our programmes.

For specific skills development, we have Writing and Phonics programmes which offer more specialised lessons.
""",
]
sglang_txt = [
    """似芙蓉涉水而来，曳曳独特；如野菊淡然绽放，暗香盈袖。汉字，你在中国这片原野上落地、生根，从此成了中国心，成了中国魂，成了我心中那段不了的汉字情。汉字，于春之暮野，我们邂逅。你向我伸出手，眼波流转，微笑蔓延。从此，我沉醉。\n\n汉字，犹记得初感受你指尖的温存，被你牵引着走向未知的远方，身后是流光飞舞的红尘岁月。在那些遥远的往事里，是你留下的快乐悲伤，惊奇瑰丽。你的故事，从春说到秋，从绿说到黄，编织着神奇的童话，承载着对世界的最初的感知。\n\n汉字，童年是你在短笛里谱写的牧歌，单纯而恬静。那些小水洼，那些烂泥巴，那些在雨天光着的小脚丫，因为有你陪伴，它们便都开成我心口的白色小花，芬芳那一颗赤诚的中国心。\n\n汉字，当我马不停蹄地向前，终于跑进成长的劫，触摸到那明媚而忧伤的青春。此时，是你，是你在我迷茫时用那闪亮的字句点燃那已微弱的心火；是你在我莫名落泪时用温柔的歌抚平我心的是疼痛；也是你教会我，在年轻的心壁上默默雕琢一种激情，一种信仰，一种向上的力量；更是你，告诉我，既然活着就要像水，点点滴滴都是真实的生命。\n\n汉字，青春是你调制的一杯鸡尾酒，眩目而浓烈。喝下那杯酒，我丢掉那些浮躁的喧嚣与空虚，和你一起或喜，或悲，或怒，或痴，任我血脉里那澎湃的中国情燃成熊熊烈火。汉字，当我满了黑发，长了腰肢，是你牵紧我的手，向那庄严的历史与人生致敬。你踏着岁月的尘埃，碾碎时间的凹凸走进历史的风雨中，我，亦无悔相随。痴迷了，沉醉了，我沦陷在你的天地里。汉字，你温柔的发梢吹来历史的夜风，讲述一个个千年古老的故事；你多情的双眸噙满泪水，赞叹着一次次文明的奇迹；你滑润的肌肤令我酥麻入心，感受着那一段段惊心动魄的传奇。五千年灿烂辉煌，在你心口浅吟低唱，余晖后的屈辱也融进你掌心的纹路，黯然神伤。微醺的""",
    """import Levenshtein\n\n# 目标文本\nA = \"目标文本\"\n\n# 模型1输出的文本\nB = \"模型1输出的文本\"\n\n# 模型2输出的文本\nC = \"模型2输出的文本\"\n\n# 计算编辑距离\ndistance_B = Levenshtein.distance(A, B)\ndistance_C = Levenshtein.distance(A, C)\n\n# 输出结果\nprint(f\"编辑距离（A与B）：{distance_B}\")\nprint(f\"编辑距离（A与C）：{distance_C}\")\n\n# 判断哪个模型更相似\nif distance_B < distance_C:\n    print(\"模型1的输出文本与目标文本更相似。\")\nelif distance_B > distance_C:\n    print(\"模型2的输出文本与目标文本更相似。\")\nelse:\n    print(\"两个模型的输出文本与目标文本同样相似。\")""",
    """Overview\n\nOur English language classes are always full of fun!\n\nESF Language uses everything from stories and role-plays to art and debate. Our English course develops students' language skills through a wide range of activities. This may include projects, speeches, and games. Hence, this instills a love of learning.\n\nThe topic-based English programmes bring books alive! Students have the opportunity to act out their favourite stories. Additionally, they may also be involved in games related to the stories. By that, this captures the imagination of our students. Plus, they will develop the four main language skills: reading, writing, speaking and listening.\n\nWhat's more, we encourage students to express themselves in English freely. Through familiar contexts, they will be communicating entirely in English. Meaning, whatever it is they do, they must do so in English. \n\nAdditionally, they get to do interesting new activities, be part of discussions, and more! On top of this, ESF offers an energetic & dynamic learning programme. We discourage sitting still on their textbooks. Instead, we incorporate games and crafts with a traditional English language course. Textbooks and a workbook only add to our programmes.\n\nFor specific skills development, we have Writing and Phonics programmes which offer more specialised lessons.""",
]
import Levenshtein

for target, trans, sgl in zip(target_txt, transfomrer_txt, sglang_txt):
    distance_tarans = Levenshtein.distance(target, trans)
    distance_sgl = Levenshtein.distance(target, sgl)
    print(f"编辑距离trans: {distance_tarans}")
    print(f"编辑距离sglang: {distance_sgl}")
