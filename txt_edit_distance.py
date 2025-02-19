target_txt = [
    """The Importance of Reading
Reading is one of the most fundamental skills a person can develop. It is not only a source of knowledge but also a gateway to imagination and creativity. In today’s fast-paced world, where technology dominates our lives, the importance of reading cannot be overstated.

First and foremost, reading enhances our knowledge. Whether it’s fiction, non-fiction, or academic texts, every book offers new information and perspectives. By reading regularly, we can learn about different cultures, historical events, scientific discoveries, and much more. This knowledge helps us become more informed citizens and better decision-makers.

Moreover, reading improves our language skills. It exposes us to new vocabulary, different writing styles, and various forms of expression. As we read, we naturally absorb the structure of sentences and the flow of ideas, which can enhance our own writing and speaking abilities. This is particularly important for students, as strong language skills are essential for academic success.

In addition to knowledge and language skills, reading also stimulates our imagination. When we read a story, we visualize the characters, settings, and events in our minds. This imaginative process fosters creativity and allows us to think outside the box. Many successful writers, artists, and innovators credit their love of reading as a key factor in their creative development.

Furthermore, reading can be a great source of relaxation and stress relief. In our busy lives, taking time to immerse ourselves in a good book can provide a much-needed escape. It allows us to disconnect from our daily worries and enter a different world, even if just for a little while. This mental break can improve our overall well-being and help us recharge.

Lastly, reading promotes empathy and understanding. Through literature, we can experience the lives and emotions of others, which helps us develop a deeper understanding of different perspectives. This is especially important in our diverse world, where empathy and compassion are crucial for fostering harmony and cooperation among people.

In conclusion, reading is an invaluable skill that enriches our lives in numerous ways. It enhances our knowledge, improves our language skills, stimulates our imagination, provides relaxation, and fosters empathy. Therefore, it is essential to cultivate a habit of reading from a young age and to encourage others to do the same. Whether it’s a novel, a magazine, or an article, every piece of reading material has the potential to inspire and educate us.""",
    """阅读的重要性
阅读是人类获取知识和信息的重要途径，也是培养思维能力和创造力的有效方式。在当今这个信息爆炸的时代，阅读的重要性愈发凸显。

首先，阅读能够拓宽我们的知识面。无论是小说、非小说还是学术论文，每一本书都蕴含着丰富的知识和不同的观点。通过阅读，我们可以了解不同的文化、历史事件、科学发现等。这些知识不仅帮助我们更好地理解世界，也使我们在日常生活中做出更明智的决策。

其次，阅读有助于提高语言能力。阅读时，我们会接触到新的词汇、不同的写作风格和多样的表达方式。这种潜移默化的学习过程，不仅能增强我们的阅读理解能力，还能提升我们的写作和口语表达能力。对于学生来说，良好的语言能力是学业成功的基础。

此外，阅读还能够激发我们的想象力。当我们阅读故事时，脑海中会浮现出人物、场景和情节。这种想象的过程不仅培养了我们的创造力，也让我们学会了从不同的角度看待问题。许多成功的作家、艺术家和创新者都将他们的阅读习惯视为创造力发展的重要因素。

阅读也是一种极好的放松方式。在快节奏的生活中，抽出时间沉浸在一本好书中，可以让我们暂时逃离现实的压力。阅读不仅能让我们享受故事的乐趣，还能帮助我们缓解焦虑，提升心理健康。

最后，阅读能够促进同理心和理解力。通过文学作品，我们可以体验他人的生活和情感，从而更深入地理解不同的观点。这在当今多元化的社会中尤为重要，因为同理心和包容心是促进人与人之间和谐相处的关键。

总之，阅读是一项极其重要的技能，它在多个方面丰富了我们的生活。它拓宽了我们的知识面，提高了语言能力，激发了想象力，提供了放松的机会，并促进了同理心的发展。因此，从小培养阅读的习惯，并鼓励他人参与阅读，是非常必要的。无论是小说、杂志还是文章，每一份阅读材料都有可能启发和教育我们。""",
    """def bubble_sort(arr):
    n = len(arr)
    # 外层循环控制比较的轮数
    for i in range(n):
        # 内层循环进行相邻元素的比较和交换
        for j in range(0, n-i-1):
            # 如果当前元素大于下一个元素，则交换它们
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# 测试代码
if __name__ == "__main__":
    sample_list = [64, 34, 25, 12, 22, 11, 90]
    sorted_list = bubble_sort(sample_list)
    print("排序后的列表:", sorted_list)""",
    """物理学第一定律通常指的是牛顿的第一运动定律，也称为惯性定律。其定义如下：

牛顿第一运动定律（惯性定律）：如果一个物体不受外力作用，或者所受的外力的合力为零，那么该物体将保持静止状态或以恒定速度沿直线运动。换句话说，物体的运动状态（静止或匀速直线运动）不会改变，除非有外力作用于它。

这个定律强调了物体的惯性，即物体抵抗运动状态改变的性质。它是经典力学的基础之一，揭示了力与运动之间的关系。""",
]
spda_txt = ["""""", """""", """""", """"""]
flash_txt = ["""""", """""", """""", """"""]
import Levenshtein

for target, spda, flash in zip(target_txt, spda_txt, flash_txt):
    distance_spda = Levenshtein.distance(target, spda)
    distance_flsh = Levenshtein.distance(target, flash)
    print(f"编辑距离spda: {distance_spda}")
    print(f"编辑距离flash: {distance_flsh}")
