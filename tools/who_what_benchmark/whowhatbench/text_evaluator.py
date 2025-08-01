from typing import Any, Union

import os
import pandas as pd
from tqdm import tqdm
from .registry import register_evaluator, BaseEvaluator
from .whowhat_metrics import TextDivergency, TextSimilarity
from .utils import patch_awq_for_inference, get_ignore_parameters_flag

default_data = {
    "en": {
        "prompts": [
            "Who is Mark Twain?",
            "Who is William Shakespeare?",
            "Who is Agatha Christie?",
            "Who is Barbara Cartland?",
            "Who is Danielle Steel?",
            "Who is Harold Robbins?",
            "Who is Georges Simenon?",
            "Who is Enid Blyton?",
            "Who is Sidney Sheldon?",
            "Who is Akira Toriyama?",
            "Who is Leo Tolstoy?",
            "Who is Alexander Pushkin?",
            "Who is Stephen King?",
            "What is C++?",
            "What is Python?",
            "What is Java?",
            "What is JavaScript?",
            "What is Perl?",
            "What is OpenCV?",
            "Who is the most famous writer?",
            "Who is the most famous inventor?",
            "Who is the most famous mathematician?",
            "Who is the most famous composer?",
            "Who is the most famous programmer?",
            "Who is the most famous athlete?",
            "Who is the most famous ancient Greek scientist?",
            "What color will you get when you mix blue and yellow?",
        ],
    },
    "cn": {
        "prompts": [
            "马克吐温是谁?",
            "谁是威廉-莎士比亚?",
            "阿加莎-克里斯蒂是谁?",
            "芭芭拉-卡特兰是谁?",
            "丹妮尔-斯蒂尔是谁?",
            "谁是哈罗德-罗宾斯?",
            "乔治-西默农是谁?",
            "伊妮德-布莱顿是谁?",
            "西德尼-谢尔顿是谁?",
            "鸟山明是谁?",
            "谁是列夫-托尔斯泰?",
            "亚历山大-普希金是谁?",
            "斯蒂芬-金是谁?",
            "C++是什么?",
            "Python是什么?",
            "什么是 Java?",
            "JavaScript是什么?",
            "什么是 Perl?",
            "什么是 OpenCV?",
            "谁是最著名的作家?",
            "谁是最有名的发明家?",
            "谁是最著名的数学家?",
            "最著名的作曲家是谁?",
            "谁是最有名的程序员?",
            "谁是最著名的运动员?",
            "谁是最著名的古希腊科学家?",
            "蓝色和黄色混合会得到什么颜色?",
        ],
    },
    "en_long": {
        "prompts": [
            """You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable".
            
            Title: Establishing Ground Truth in Fake News Detection: Methods and Challenges
            
            Abstract:
            The proliferation of misinformation on digital platforms has amplified the need for robust fake news detection systems. Central to developing and evaluating such systems is the establishment of a reliable ground truth. This paper reviews methodologies used to determine the veracity of news content, focusing on expert annotation, fact-checking services, and crowdsourcing. It also discusses challenges such as bias, subjectivity, and the evolving nature of truth in socio-political contexts.
            
            1. Introduction
            Fake news—deliberately false or misleading information presented as news—has become a global concern, impacting elections, public health, and societal trust. Machine learning and natural language processing (NLP) techniques have emerged to detect fake news, but these systems require labeled datasets with a ground truth indicating whether content is true or false. Defining and verifying this ground truth is complex due to subjective interpretation, ideological bias, and temporal shifts in facts.
            
            2. Ground Truth Establishment Methods
            2.1 Expert Annotation
            Many datasets rely on professional journalists, domain experts, or researchers to manually annotate articles. Annotators verify claims against authoritative sources and follow defined guidelines to classify news as true, partially true, or false. This method ensures high-quality labels but is costly and time-consuming.
            2.2 Fact-Checking Organizations
            Ground truth is often derived from fact-checking platforms such as PolitiFact, Snopes, and FactCheck.org. These organizations apply rigorous verification processes and publish verdicts on individual claims. Datasets like LIAR and FakeNewsNet incorporate these verdicts as labels. While credible, such sources may still exhibit systemic biases based on their editorial standards.
            2.3 Crowdsourcing
            Some studies use non-expert crowdsourced workers to annotate news, leveraging redundancy and aggregation to improve label reliability. However, crowd workers may lack the domain knowledge or context needed to accurately judge factual correctness, leading to inconsistent labels.
            2.4 Source Credibility and Heuristics
            In some automated settings, the credibility of the news source is used as a proxy for truth. For example, mainstream media might be labeled as reliable, while fringe websites are flagged as less trustworthy. This approach is scalable but oversimplifies the nuanced nature of fake news.
            
            3. Challenges and Limitations
            Bias and Subjectivity: Annotator beliefs and the framing of news can influence labeling decisions.
            Temporal Dynamics: What is false today might become true tomorrow, and vice versa, particularly in fast-evolving domains like science or politics.
            Lack of Consensus: Multiple fact-checkers may disagree on the same piece of content.
            Granularity of Labels: Truth is often not binary; claims can be partially true or require additional context.
            
            4. Conclusion
            Establishing ground truth for fake news remains a critical and non-trivial task. Despite the availability of expert fact-checking and annotation methodologies, challenges such as bias, subjectivity, and the dynamic nature of truth persist. Future work should focus on consensus-building, context-aware labeling, and hybrid annotation pipelines combining expert and crowd-based approaches.
            
            Question: How is the ground truth for fake news established?""",
            
            """Summarize the dialogue into a few short sentences. The following are some examples.
            Dialogue:
            
            Emma: hey did you by any chance grab my water bottle after yoga?
            
            Maya: hmm wait let me check my bag…
            Maya: omg yes I totally did
            Emma: 😑
            Maya: I swear I thought it was mine!! they look identical
            Emma: I figured lol. mine has the little dent at the bottom
            Maya: yup just noticed that
            Emma: could you bring it to class tomorrow? I’ve been dying of thirst all day 😩
            Maya: YES ofc. I’ll bring it. sorry for kidnapping your hydration vessel 😬
            Emma: unacceptable
            Maya: I’ll write it an apology letter
            
            Emma: also btw did you understand anything from Dr. Chen’s lecture today?
            Maya: not even a little. I zoned out after the first 10 minutes
            Emma: right?? like what was even going on with that integral at the end
            Maya: he just… kept writing symbols
            Emma: I’m gonna try watching his old lecture again tonight. maybe it’ll help
            Maya: let me know if it does. otherwise we may need to form a survival group
            
            Emma: anyway, are you still going to Maya’s thing on Saturday?
            Maya: umm yes, I think so. depends on if I finish my project
            Emma: the one for the art class?
            Maya: yep. I’m like halfway done, but the clay sculpture keeps collapsing 🙃
            Emma: rude
            Maya: very. gravity is NOT on my side this week
            Emma: lmk if you need help. I have some of that wire mesh stuff left over from last semester
            Maya: omg that might actually save me. you’re a hero
            Emma: I do what I can 😌
            Maya: ok I’ll come by your place tomorrow after class if that’s cool?
            Emma: yeah just text me when you’re outside
            Maya: will do. also promise not to steal anything else of yours
            Emma: thank you. one theft per semester is my limit
            Maya: noted 😂
            
            Summary:""",
            
            """Please determine the type of the question below. Here are some examples of questions.
            Question: Who invented the process of pasteurization?
            Type: Individual
            
            Question: What was the purpose of the Marshall Plan?
            Type: Reason
            
            Question: How much does a 1921 Morgan silver dollar cost today?
            Type: Price
            
            Question: What are the three branches of the U.S. government?
            Type: Group or organization of person
            
            Question: What makes the hummingbird unique among birds?
            Type: Description of something
            
            Question: What is the capital of Burkina Faso?
            Type: Other entity
            
            Question: What book features a character named Atticus Finch?
            Type: Invention, book and other creative piece
            
            Question: How many people died in the Great Fire of London?
            Type: Number of something
            
            Question: Who was the President of the Confederate States during the Civil War?
            Type: Individual
            
            Question: What is quantum entanglement?
            Type: Definition of something
            
            Question: Why do whales beach themselves?
            Type: Reason
            
            Question: What comic strip was created by Charles Schulz?
            Type: Invention, book and other creative piece
            
            Question: How many minutes are in a decade?
            Type: Number of something
            
            Question: What was the slogan of the first Apple iPhone advertisement?
            Type: Description of something
            
            Question: Who were the original members of the band Queen?
            Type: Group or organization of person
            
            Question: What is the meaning of the word “ephemeral”?
            Type: Definition of something
            
            Question: Who designed the Eiffel Tower?
            Type: Individual
            
            Question: How many moons does Jupiter have?
            Type: Number of something
            
            Question: What was the original name of New York City?
            Type:""",
            
            """Answer the question based on the given passages. Only give me the answer and do not output any other words.
            Passage 1:
            Brown v. Board of Education
            Brown v. Board of Education, 347 U.S. 483 (1954), was a landmark United States Supreme Court case in which the Court declared state laws establishing separate public schools for black and white students to be unconstitutional. The decision overturned the "separate but equal" doctrine established by Plessy v. Ferguson in 1896 and helped dismantle racial segregation across the United States.
            
            Background
            The case was a consolidation of five different cases all challenging the constitutionality of racial segregation in public schools. African American students had been denied admission to certain public schools based on laws allowing segregation by race. The lead plaintiff, Oliver Brown, filed suit against the Board of Education of Topeka, Kansas, after his daughter was denied entry to an all-white school near their home.
            
            Decision
            The Supreme Court, led by Chief Justice Earl Warren, unanimously ruled that "separate educational facilities are inherently unequal," violating the Equal Protection Clause of the Fourteenth Amendment. The ruling mandated the desegregation of schools across America, though implementation met with resistance and took many years.
            
            Passage 2:
            Roe v. Wade
            Roe v. Wade, 410 U.S. 113 (1973), was a landmark decision of the United States Supreme Court in which the Court ruled that a state law that banned abortions except to save the life of the mother was unconstitutional under the Due Process Clause of the Fourteenth Amendment. The ruling effectively legalized abortion nationwide and ignited decades of political and legal debate.
            
            Background
            The case was brought by "Jane Roe" (a pseudonym for Norma McCorvey), who challenged the Texas law criminalizing most abortions. Roe argued that the law violated her constitutional rights. The case was heard during the 1971-1972 term and decided in 1973.
            
            Decision
            The Court held that a woman's right to privacy included the decision to have an abortion but balanced this against the state's interests in regulating abortions and protecting prenatal life. The ruling created a trimester framework limiting state intervention early in pregnancy.
            
            Question: Which case was decided first, Brown v. Board of Education or Roe v. Wade?""",
            
            """Answer the question based on the story as concisely as you can, using a single phrase if possible.
            Liam had never known the bustling comfort of a large household. From the earliest days he could recall, the modest cottage at the edge of Ashwood village had been his only home. The worn wooden door, its paint peeling from years of rain and sun, opened to a small sitting room where the scent of old books mingled with the faint aroma of lavender and fresh bread. This humble abode belonged to his grandmother, a stern yet kind woman whose hands told stories of decades spent in toil and tender care.
            
            Though the cottage was simple—furnished with a creaking oak table, a patchwork quilt upon the narrow bed, and shelves lined with knickknacks collected through the years—it held a quiet warmth that comforted Liam against the chill of the outside world. His parents, long gone, had entrusted him to her care when he was but a child, and since then, the two had forged a life marked by routine and resilience.
            
            Each morning, Liam awoke to the soft creak of his grandmother’s footsteps in the kitchen, where a pot of tea simmered and fresh bread awaited. Though the village was small and the neighbors few, Liam found solace in the steady rhythm of his days and the steadfast presence of the woman who had become both guardian and guide. It was a life shaped less by abundance and more by the quiet strength of kinship and simple comforts.
            Question:  What is Liam's living situation?""",
            
            """Answer the following question based on the above text, only give me the answer and do not output any other words.
            
            The room was dimly lit by the fading afternoon sun filtering through the lace curtains, casting delicate shadows upon the worn velvet armchair where Clara sat, her fingers nervously twisting the edge of a handkerchief. Outside, the garden lay silent, the rose bushes heavy with dew and the faint hum of bees long departed. The grandfather clock in the hallway chimed softly, marking the hour with a slow, measured toll that seemed to echo the heaviness in the air. Though her lips formed a faint smile, her eyes betrayed a restless uncertainty that no outward calm could conceal.
            
            “You must understand,” she murmured, “that it is not simply a matter of choice, but of circumstance. My father’s debts have weighed heavily upon us, and the suitor’s offer is, for all its faults, a promise of security.” She glanced briefly at the portrait of her late mother, whose serene expression seemed to judge the scene before her with silent disappointment. Her companion nodded slowly, sensing the storm beneath her calm exterior. He shifted in his chair, the worn leather creaking faintly, and gestured toward the heavy oak bookcase lined with dusty volumes—tales of adventure and lost fortunes that seemed strangely out of place in this somber moment.
            
            “And yet,” he said softly, “do you truly wish to accept such a fate?” The question hung in the air, mingling with the faint scent of lavender and the distant tolling of church bells. Clara’s fingers tightened around the handkerchief as she weighed the promise of safety against the shadow of resignation that it cast. Outside, the sky deepened into twilight, and a single star began to twinkle, indifferent to the choices that troubled the heart within.
            
            Question: What is Clara’s primary concern about her suitor’s offer?""",
            
            """Answer the question based on the given passages.
            Passage 1:
            Isabella of France
            Isabella of France (1295–1358), daughter of King Philip IV of France and Joan I of Navarre, ascended to the position of queen consort of England through her politically motivated union with King Edward II. This marriage, orchestrated amidst complex Franco-English relations, was intended to fortify a fragile peace between the two kingdoms. Isabella's legacy is notably marked by her instrumental role in the eventual deposition of her husband, an act that earned her the sobriquet "She-Wolf of France," reflecting both her formidable political resolve and the controversy surrounding her ambitions within the English court.
            
            Passage 2:
            Margaret Cavendish
            Margaret Cavendish, Duchess of Newcastle-upon-Tyne (1623–1673), born Margaret Lucas into a family of modest noble standing, emerged as an extraordinarily prolific writer, philosopher, and natural scientist during a period when such pursuits were rarely encouraged among women of her rank. Her marriage to William Cavendish, 1st Duke of Newcastle, granted her access to intellectual circles where she produced an extensive oeuvre encompassing poetry, theatrical works, and philosophical treatises. Despite facing societal skepticism and critique, her unconventional perspectives challenged prevailing norms on gender and scientific inquiry, making her a singular figure in seventeenth-century English letters.
            
            Passage 3:
            Anne Boleyn
            Anne Boleyn (c. 1501–1536), scion of the English nobility and daughter of Thomas Boleyn, Earl of Wiltshire and Ormond, and Lady Elizabeth Howard, spent her formative years partially abroad in the courts of the Netherlands and France, an experience that shaped her cultural sophistication and political acumen. Her eventual marriage to King Henry VIII was fraught with religious and political upheaval, catalyzing the English Reformation and the severance from papal authority. Anne's tragic downfall, culminating in charges of treason, adultery, and incest—charges historians continue to scrutinize—sealed her fate and profoundly altered the course of English history.
            
            Question: From which region or court did the wife of King Henry VIII originate during her youth?""",
            
            """Answer the question based on the given passage:
            Veil of Shadows is a 2018 neo-noir thriller directed by Lucas Renaud, featuring a chilling score composed by Jóhann Jóhannsson. The soundtrack combines minimalist orchestral arrangements with electronic soundscapes to heighten the film’s tense and brooding atmosphere.
            
            Plot
            Set in a rain-soaked metropolis, the story follows detective Marcus Hale (played by Oscar Isaac) as he investigates a series of mysterious disappearances linked to a secretive underground society. As Marcus delves deeper, he forms an uneasy alliance with Evelyn Drake (Tessa Thompson), a journalist whose own past is entwined with the case. Together, they unravel a web of corruption and betrayal that threatens to consume them both.
            
            Cast
            Oscar Isaac as Detective Marcus Hale
            Tessa Thompson as Evelyn Drake
            Mark Strong as Chief Inspector Graves
            Ruth Negga as Lydia Cross
            Ben Mendelsohn as Victor Kane
            
            Reception
            The film received generally positive reviews, with critics lauding its atmospheric direction and compelling performances. The Guardian praised Jóhannsson’s score as “a haunting masterpiece that perfectly complements the film’s dark tone.” However, some reviewers found the plot overly intricate, noting that it occasionally sacrificed clarity for mood. Despite this, Veil of Shadows was nominated for several awards, including Best Cinematography and Best Original Score.
            
            Question: Who composed the score for Veil of Shadows?""",
            
            """You are given a report by a government agency. Write a one-page summary of the report.
            
            Report:
            This report provides a comprehensive analysis of the current state and effectiveness of the nation’s public transportation infrastructure. It evaluates various modes of transport, including buses, trains, subways, and light rail systems, across urban and rural regions. The report highlights key challenges such as aging infrastructure, funding shortfalls, and service gaps that hinder accessibility and efficiency.
            
            Data collected over the past five years reveal a steady increase in ridership in metropolitan areas, reflecting growing demand for sustainable and affordable transit options. However, rural areas continue to experience limited service, leading to transportation inequities that affect economic opportunities and quality of life for residents.
            
            The report examines the impact of recent federal and state investments aimed at modernizing transit fleets, improving safety standards, and integrating emerging technologies such as real-time tracking and contactless payment systems. These initiatives have shown promise in enhancing rider experience and operational efficiency, though they require ongoing funding and coordinated planning.
            
            Environmental considerations are a significant focus, with recommendations to expand electric and hybrid vehicle adoption to reduce greenhouse gas emissions and improve air quality. Additionally, the report emphasizes the importance of inclusive planning that addresses the needs of elderly, disabled, and low-income populations.
            
            In conclusion, the report calls for increased multi-level government collaboration, innovative financing mechanisms, and community engagement to ensure that public transportation systems are resilient, equitable, and capable of meeting future demands. It stresses that investment in public transit is critical not only for mobility but also for economic growth, environmental sustainability, and social inclusion.
            Now, write a one-page summary of the report.
            
            Summary:""",
            
            """You are given several news passages. Write a short summary of all news.
            News:
            Passage 1:
            Global Climate Summit Concludes with New Agreements
            The 2025 Global Climate Summit, held over two weeks in Geneva, brought together leaders from nearly 200 countries to address the escalating challenges posed by climate change. After intense negotiations, the summit concluded with a landmark agreement aimed at reducing global carbon emissions by 50% by 2035 compared to 2020 levels. Key elements of the deal include a phased elimination of coal-fired power plants, increased subsidies and incentives for renewable energy projects such as solar and wind, and a commitment by wealthier nations to provide $200 billion annually in climate adaptation and mitigation aid to developing countries. The summit also emphasized the importance of protecting biodiversity and investing in sustainable agriculture. While the agreement has been hailed by many environmental organizations as a crucial step forward, some activists remain skeptical, citing a lack of binding enforcement mechanisms and concerns over the pace of implementation.
            
            Passage 2:
            Tech Giant Announces Breakthrough in Battery Technology
            Voltatek, a global leader in clean technology innovation, has announced a major breakthrough in battery technology that could dramatically reshape the electric vehicle (EV) and consumer electronics industries. The company’s new lithium-silicon composite battery reportedly doubles the energy density compared to conventional lithium-ion cells while simultaneously reducing charging times by 50%. This advancement is expected to significantly extend the driving range of EVs and decrease wait times at charging stations, addressing two of the most critical barriers to wider EV adoption. Voltatek’s CEO highlighted that the new batteries also use more sustainable materials and are easier to recycle, reducing environmental impact. Commercial production is slated to begin in late 2027, with partnerships already secured with several major automotive manufacturers. Industry analysts predict this innovation could accelerate the global transition to greener transportation and power solutions.
            
            Passage 3:
            International Space Mission Successfully Launches
            Yesterday marked a historic day for space exploration as the multinational Artemis II mission launched successfully from Cape Canaveral. The spacecraft, carrying four astronauts representing the United States, Canada, Japan, and Germany, will orbit the Moon over the course of a 10-day mission designed to test new life-support systems and conduct scientific experiments related to lunar geology and radiation exposure. Artemis II is the first crewed lunar mission in over half a century and serves as a critical precursor to Artemis III, which aims to land humans on the lunar surface by 2026. NASA Administrator emphasized that this mission symbolizes international cooperation in space and is a vital step towards the long-term goal of establishing a sustainable human presence on the Moon and eventually sending crewed missions to Mars. Public excitement has been high, with millions tuning in for the live broadcast of the launch and associated educational outreach programs inspiring the next generation of scientists and engineers.
            
            Now, write a summary of all the news.
            Summary:""",
            
            """Answer the question based on the given passage.
            
            Passage:
            The History and Influence of Impressionism
            
            Impressionism, emerging in the late 19th century, was a revolutionary art movement that challenged traditional academic painting in France. Its pioneers, including Claude Monet, Pierre-Auguste Renoir, and Camille Pissarro, emphasized capturing the transient effects of light and color rather than meticulous detail. The movement’s name derives from Monet’s painting Impression, Sunrise exhibited in 1874, which critics initially mocked but later embraced as emblematic of a new artistic vision.
            
            The Impressionists frequently painted en plein air (outdoors), striving to portray scenes of modern life, landscapes, and fleeting moments with loose brushwork and vibrant palettes. Their approach broke away from the somber tones and historical subjects favored by the Salon, the official art exhibition of the Académie des Beaux-Arts.
            
            Despite early criticism and rejection by traditional art institutions, Impressionism profoundly influenced subsequent movements like Post-Impressionism and Fauvism. Moreover, it democratized art by focusing on everyday subjects and ordinary people rather than aristocratic or mythological themes.
            
            Beyond painting, Impressionism’s impact extended into literature and music, inspiring writers such as Marcel Proust and composers like Claude Debussy, who echoed its emphasis on atmosphere and mood.
            
            Passage:
            The Rise and Global Impact of Streaming Services on the Film Industry
            
            In the early 21st century, streaming platforms such as Netflix, Amazon Prime Video, and later Disney+, fundamentally transformed the film and television industry. Initially starting as a DVD-by-mail service, Netflix pivoted to streaming in 2007, leading a digital revolution that disrupted the traditional distribution models of cinema and broadcast television.
            
            One of the most significant shifts caused by streaming has been the collapse of the theatrical release window. Previously, films would debut exclusively in cinemas for months before becoming available for home viewing. Streaming services, however, began offering original films directly to subscribers, sometimes releasing them simultaneously in theaters and online—most notably during the COVID-19 pandemic, which accelerated this trend.
            
            This shift has sparked debates across the industry. While some filmmakers argue that streaming democratizes access and offers greater creative freedom, others worry it diminishes the cultural experience of cinema and undermines box office revenue. Additionally, streaming platforms use complex algorithms to personalize recommendations, influencing viewer habits and even greenlighting content based on data-driven predictions rather than purely artistic vision.
            
            Globally, the reach of streaming has enabled regional and independent films to gain international exposure. Korean dramas, Spanish thrillers, and Indian series now enjoy mainstream success beyond their borders, contributing to a diversification of storytelling and audience tastes worldwide.
            
            Passage:
            Largest Automakers in the World 2025 by Production Volume
            According to industry reports, the top three largest automobile manufacturers by production volume in 2025 are:
            Toyota Motor Corporation - Producing approximately 10.5 million vehicles annually, Toyota remains the largest automaker globally. Headquartered in Japan, Toyota is known for its focus on hybrid and fuel-efficient vehicles.
            Volkswagen Group - Producing about 9.8 million vehicles, Volkswagen maintains a strong global presence with multiple brands including Audi, Porsche, and Skoda.
            General Motors - With a production volume of around 6.5 million vehicles, GM is the largest American automaker, emphasizing electric vehicle development in recent years.
            
            Question: In what ways have streaming platforms altered both the distribution of films and the global cultural landscape of entertainment?
            Answer:""",
            
            """There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?
            
            Paragraph 1: In 2005, Sega partnered with Atmos Tokyo to produce sports-themed iDogs as part of a new line, titled "iDog x Atmos". In 2007, Tiger Electronics released the iDog Amp'd, an upgraded version of the iDog with stereo speakers that nod its head and tap its foot in time with the music being played.  The iDog Pup, a puppy version of the iDog with poseable ears and a moving head, was released in 2007 and was a localized version of the iDog Mini released in Japan in 2005. Also in 2007, two variants called "SpiDogs" were released to promote the film Spider-Man 3. They resemble the standard Spider-Man costume and the Black Suit respectively. In 2008, three new versions of the iDog were released. The iDog Clip is a fully functional mini-sized iDog which can be clipped onto a backpack, the iDog Dance is a larger version of the iDog that stands up and dances in time to the music, it also features 7 touch sensors (like the iCat), and the iDog Soft Speaker, a plush version of the iDog Amp'd that remixes some songs on it and has a pocket where the battery compartment is so you can store things in it. In 2009, as one of the last iDogs released, Hasbro released the iDog Plush Puppy. This smaller plush iDog came in pink and purple and was one of the first iDogs that didn't need batteries. It came with a cord attached to it, like the iDog Clip. But it had the light patterns stitched onto it.
            
            Paragraph 2:
            The iDog Clip, introduced in 2008, was a compact version of the original iDog that users could attach to backpacks or clothing. Despite its smaller size, it retained the full functionality of the larger models and responded to music and touch input. It proved especially popular among younger children due to its portability and playful design.
            
            Paragraph 3:
            The Colossal Kongs, a tag team consisting of "King Kong" and "Awesome Kong", were active in the early 1990s. They performed in multiple wrestling promotions, including WCW and the Global Wrestling Federation. Though they never captured a major championship, they were known for their size and theatrics.
            
            Paragraph 4:
            The iDog Clip, introduced in 2008, was a compact version of the original iDog that users could attach to backpacks or clothing. Despite its smaller size, it retained the full functionality of the larger models and responded to music and touch input. It proved especially popular among younger children due to its portability and playful design.
            
            Paragraph 5:
            In 1993, the wrestling team of King Kong and Awesome Kong made their WCW debut. Managed by Harley Race, they participated in notable events such as Clash of the Champions XXIV and Starrcade. Although they did not win, their large frames and masked appearances gained attention.
            
            Please enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.
            
            The final answer is:""",
            
            """Here are 5 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.
            
            Paragraph 1:
            In 1986, the Soviet Union launched the Mir space station, which became one of the most ambitious long-duration human spaceflight projects of the 20th century. It consisted of a core module and later added several research modules developed by various Soviet research institutes. The station hosted over 100 different people from 12 countries, including collaborative missions with the United States under the Shuttle-Mir program. Mir remained in orbit for 15 years before being deorbited in 2001. Its legacy influenced the design and international cooperation model used for the International Space Station.
            
            Paragraph 2:
            Ada Lovelace is often regarded as the first computer programmer due to her work on Charles Babbage’s Analytical Engine. In 1843, she translated an Italian article about the Engine and added a comprehensive set of notes, including a method for calculating Bernoulli numbers—considered the first algorithm intended for a machine. Her visionary understanding of the machine’s capabilities went beyond mere calculation, suggesting it could manipulate symbols and create music. Her contributions were largely unrecognized during her lifetime but gained prominence in the late 20th century.
            
            Paragraph 3:
            The Great Fire of London in 1666 was a catastrophic event that destroyed a large part of the city, including 87 churches and around 13,000 houses. Starting in a bakery on Pudding Lane, the fire spread rapidly due to strong winds and the use of timber in construction. Although only six deaths were officially recorded, it is believed the real number was higher. The fire led to major changes in building regulations and urban planning, with notable efforts led by Christopher Wren to redesign London’s skyline.
            
            Paragraph 4:
            The Galápagos Islands, located off the coast of Ecuador, are famous for their unique biodiversity and significant influence on Charles Darwin’s theory of evolution. The archipelago consists of 13 main islands and several smaller ones, each hosting distinct ecosystems. Species like the marine iguana, Galápagos tortoise, and finches exhibit striking adaptations. Strict conservation policies are now in place due to the islands’ ecological importance and vulnerability to tourism and climate change.
            
            Paragraph 5:
            Marie Curie remains one of the most celebrated scientists in history. She was the first woman to win a Nobel Prize and the only person to win in two different scientific fields: Physics and Chemistry. Alongside her husband Pierre, she discovered the elements polonium and radium, which led to the development of radiotherapy for cancer. Curie’s research was groundbreaking but exposed her to high levels of radiation, which eventually led to her death. Her notebooks are still radioactive today.
            
            Abstract:
            This passage details a historic urban disaster that began in a bakery and rapidly consumed thousands of homes and many churches. It significantly reshaped the city’s physical structure and prompted major urban planning reforms, spearheaded by a well-known architect.
            
            Please enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.
            The answer is:""",
            
            """Please complete the code given below.
            
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import classification_report, confusion_matrix
            
            # Load dataset
            df = pd.read_csv('heart.csv')
            
            # Check for missing values
            if df.isnull().sum().sum() > 0:
                df = df.dropna()
            
            # Define features and target
            X = df.drop('target', axis=1)
            y = df['target']
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Standardize the feature columns
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Initialize the model
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Evaluate the model
            
            Next line of code:""",
            
            """You are given a scientific article and a question. Respond to the query as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable".
            
            Title: Establishing Ground Truth in Fake News Detection: Methods and Challenges
            
            Abstract:
            The proliferation of misinformation on digital platforms has amplified the need for robust fake news detection systems. Central to developing and evaluating such systems is the establishment of a reliable ground truth. This paper reviews methodologies used to determine the veracity of news content, focusing on expert annotation, fact-checking services, and crowdsourcing. It also discusses challenges such as bias, subjectivity, and the evolving nature of truth in socio-political contexts.
            
            1. Introduction
            Fake news—deliberately false or misleading information presented as news—has become a global concern, impacting elections, public health, and societal trust. Machine learning and natural language processing (NLP) techniques have emerged to detect fake news, but these systems require labeled datasets with a ground truth indicating whether content is true or false. Defining and verifying this ground truth is complex due to subjective interpretation, ideological bias, and temporal shifts in facts.
            
            2. Ground Truth Establishment Methods
            2.1 Expert Annotation
            Many datasets rely on professional journalists, domain experts, or researchers to manually annotate articles. Annotators verify claims against authoritative sources and follow defined guidelines to classify news as true, partially true, or false. This method ensures high-quality labels but is costly and time-consuming.
            2.2 Fact-Checking Organizations
            Ground truth is often derived from fact-checking platforms such as PolitiFact, Snopes, and FactCheck.org. These organizations apply rigorous verification processes and publish verdicts on individual claims. Datasets like LIAR and FakeNewsNet incorporate these verdicts as labels. While credible, such sources may still exhibit systemic biases based on their editorial standards.
            2.3 Crowdsourcing
            Some studies use non-expert crowdsourced workers to annotate news, leveraging redundancy and aggregation to improve label reliability. However, crowd workers may lack the domain knowledge or context needed to accurately judge factual correctness, leading to inconsistent labels.
            2.4 Source Credibility and Heuristics
            In some automated settings, the credibility of the news source is used as a proxy for truth. For example, mainstream media might be labeled as reliable, while fringe websites are flagged as less trustworthy. This approach is scalable but oversimplifies the nuanced nature of fake news.
            
            3. Challenges and Limitations
            Bias and Subjectivity: Annotator beliefs and the framing of news can influence labeling decisions.
            Temporal Dynamics: What is false today might become true tomorrow, and vice versa, particularly in fast-evolving domains like science or politics.
            Lack of Consensus: Multiple fact-checkers may disagree on the same piece of content.
            Granularity of Labels: Truth is often not binary; claims can be partially true or require additional context.
            
            4. Conclusion
            Establishing ground truth for fake news remains a critical and non-trivial task. Despite the availability of expert fact-checking and annotation methodologies, challenges such as bias, subjectivity, and the dynamic nature of truth persist. Future work should focus on consensus-building, context-aware labeling, and hybrid annotation pipelines combining expert and crowd-based approaches.
            
            Question: How is the ground truth for fake news established?""",
            
            """Write a brief summary the dialogue into a few short sentences. The following are some examples.
            Dialogue:
            
            Emma: hey did you by any chance grab my water bottle after yoga?
            
            Maya: hmm wait let me check my bag…
            Maya: omg yes I totally did
            Emma: 😑
            Maya: I swear I thought it was mine!! they look identical
            Emma: I figured lol. mine has the little dent at the bottom
            Maya: yup just noticed that
            Emma: could you bring it to class tomorrow? I’ve been dying of thirst all day 😩
            Maya: YES ofc. I’ll bring it. sorry for kidnapping your hydration vessel 😬
            Emma: unacceptable
            Maya: I’ll write it an apology letter
            
            Emma: also btw did you understand anything from Dr. Chen’s lecture today?
            Maya: not even a little. I zoned out after the first 10 minutes
            Emma: right?? like what was even going on with that integral at the end
            Maya: he just… kept writing symbols
            Emma: I’m gonna try watching his old lecture again tonight. maybe it’ll help
            Maya: let me know if it does. otherwise we may need to form a survival group
            
            Emma: anyway, are you still going to Maya’s thing on Saturday?
            Maya: umm yes, I think so. depends on if I finish my project
            Emma: the one for the art class?
            Maya: yep. I’m like halfway done, but the clay sculpture keeps collapsing 🙃
            Emma: rude
            Maya: very. gravity is NOT on my side this week
            Emma: lmk if you need help. I have some of that wire mesh stuff left over from last semester
            Maya: omg that might actually save me. you’re a hero
            Emma: I do what I can 😌
            Maya: ok I’ll come by your place tomorrow after class if that’s cool?
            Emma: yeah just text me when you’re outside
            Maya: will do. also promise not to steal anything else of yours
            Emma: thank you. one theft per semester is my limit
            Maya: noted 😂
            
            Summary:""",
            
            """Respond to the query based on the given passages. Only give me the answer and do not output any other words.
            Passage 1:
            Brown v. Board of Education
            Brown v. Board of Education, 347 U.S. 483 (1954), was a landmark United States Supreme Court case in which the Court declared state laws establishing separate public schools for black and white students to be unconstitutional. The decision overturned the "separate but equal" doctrine established by Plessy v. Ferguson in 1896 and helped dismantle racial segregation across the United States.
            
            Background
            The case was a consolidation of five different cases all challenging the constitutionality of racial segregation in public schools. African American students had been denied admission to certain public schools based on laws allowing segregation by race. The lead plaintiff, Oliver Brown, filed suit against the Board of Education of Topeka, Kansas, after his daughter was denied entry to an all-white school near their home.
            
            Decision
            The Supreme Court, led by Chief Justice Earl Warren, unanimously ruled that "separate educational facilities are inherently unequal," violating the Equal Protection Clause of the Fourteenth Amendment. The ruling mandated the desegregation of schools across America, though implementation met with resistance and took many years.
            
            Passage 2:
            Roe v. Wade
            Roe v. Wade, 410 U.S. 113 (1973), was a landmark decision of the United States Supreme Court in which the Court ruled that a state law that banned abortions except to save the life of the mother was unconstitutional under the Due Process Clause of the Fourteenth Amendment. The ruling effectively legalized abortion nationwide and ignited decades of political and legal debate.
            
            Background
            The case was brought by "Jane Roe" (a pseudonym for Norma McCorvey), who challenged the Texas law criminalizing most abortions. Roe argued that the law violated her constitutional rights. The case was heard during the 1971-1972 term and decided in 1973.
            
            Decision
            The Court held that a woman's right to privacy included the decision to have an abortion but balanced this against the state's interests in regulating abortions and protecting prenatal life. The ruling created a trimester framework limiting state intervention early in pregnancy.
            
            Question: Which case was decided first, Brown v. Board of Education or Roe v. Wade?""",
            
            """Respond to the query based on the story as concisely as you can, using a single phrase if possible.
            Liam had never known the bustling comfort of a large household. From the earliest days he could recall, the modest cottage at the edge of Ashwood village had been his only home. The worn wooden door, its paint peeling from years of rain and sun, opened to a small sitting room where the scent of old books mingled with the faint aroma of lavender and fresh bread. This humble abode belonged to his grandmother, a stern yet kind woman whose hands told stories of decades spent in toil and tender care.
            
            Though the cottage was simple—furnished with a creaking oak table, a patchwork quilt upon the narrow bed, and shelves lined with knickknacks collected through the years—it held a quiet warmth that comforted Liam against the chill of the outside world. His parents, long gone, had entrusted him to her care when he was but a child, and since then, the two had forged a life marked by routine and resilience.
            
            Each morning, Liam awoke to the soft creak of his grandmother’s footsteps in the kitchen, where a pot of tea simmered and fresh bread awaited. Though the village was small and the neighbors few, Liam found solace in the steady rhythm of his days and the steadfast presence of the woman who had become both guardian and guide. It was a life shaped less by abundance and more by the quiet strength of kinship and simple comforts.
            Question:  What is Liam's living situation?""",
            
            """Respond to the query based on the given passages.
            Passage 1:
            Isabella of France
            Isabella of France (1295–1358), daughter of King Philip IV of France and Joan I of Navarre, ascended to the position of queen consort of England through her politically motivated union with King Edward II. This marriage, orchestrated amidst complex Franco-English relations, was intended to fortify a fragile peace between the two kingdoms. Isabella's legacy is notably marked by her instrumental role in the eventual deposition of her husband, an act that earned her the sobriquet "She-Wolf of France," reflecting both her formidable political resolve and the controversy surrounding her ambitions within the English court.
            
            Passage 2:
            Margaret Cavendish
            Margaret Cavendish, Duchess of Newcastle-upon-Tyne (1623–1673), born Margaret Lucas into a family of modest noble standing, emerged as an extraordinarily prolific writer, philosopher, and natural scientist during a period when such pursuits were rarely encouraged among women of her rank. Her marriage to William Cavendish, 1st Duke of Newcastle, granted her access to intellectual circles where she produced an extensive oeuvre encompassing poetry, theatrical works, and philosophical treatises. Despite facing societal skepticism and critique, her unconventional perspectives challenged prevailing norms on gender and scientific inquiry, making her a singular figure in seventeenth-century English letters.
            
            Passage 3:
            Anne Boleyn
            Anne Boleyn (c. 1501–1536), scion of the English nobility and daughter of Thomas Boleyn, Earl of Wiltshire and Ormond, and Lady Elizabeth Howard, spent her formative years partially abroad in the courts of the Netherlands and France, an experience that shaped her cultural sophistication and political acumen. Her eventual marriage to King Henry VIII was fraught with religious and political upheaval, catalyzing the English Reformation and the severance from papal authority. Anne's tragic downfall, culminating in charges of treason, adultery, and incest—charges historians continue to scrutinize—sealed her fate and profoundly altered the course of English history.
            
            Question: From which region or court did the wife of King Henry VIII originate during her youth?""",
            
            """Respond to the query based on the given passage:
            Veil of Shadows is a 2018 neo-noir thriller directed by Lucas Renaud, featuring a chilling score composed by Jóhann Jóhannsson. The soundtrack combines minimalist orchestral arrangements with electronic soundscapes to heighten the film’s tense and brooding atmosphere.
            
            Plot
            Set in a rain-soaked metropolis, the story follows detective Marcus Hale (played by Oscar Isaac) as he investigates a series of mysterious disappearances linked to a secretive underground society. As Marcus delves deeper, he forms an uneasy alliance with Evelyn Drake (Tessa Thompson), a journalist whose own past is entwined with the case. Together, they unravel a web of corruption and betrayal that threatens to consume them both.
            
            Cast
            Oscar Isaac as Detective Marcus Hale
            Tessa Thompson as Evelyn Drake
            Mark Strong as Chief Inspector Graves
            Ruth Negga as Lydia Cross
            Ben Mendelsohn as Victor Kane
            
            Reception
            The film received generally positive reviews, with critics lauding its atmospheric direction and compelling performances. The Guardian praised Jóhannsson’s score as “a haunting masterpiece that perfectly complements the film’s dark tone.” However, some reviewers found the plot overly intricate, noting that it occasionally sacrificed clarity for mood. Despite this, Veil of Shadows was nominated for several awards, including Best Cinematography and Best Original Score.
            
            Question: Who composed the score for Veil of Shadows?""",
            
            """Respond to the query based on the given passage.
            
            Passage:
            The History and Influence of Impressionism
            
            Impressionism, emerging in the late 19th century, was a revolutionary art movement that challenged traditional academic painting in France. Its pioneers, including Claude Monet, Pierre-Auguste Renoir, and Camille Pissarro, emphasized capturing the transient effects of light and color rather than meticulous detail. The movement’s name derives from Monet’s painting Impression, Sunrise exhibited in 1874, which critics initially mocked but later embraced as emblematic of a new artistic vision.
            
            The Impressionists frequently painted en plein air (outdoors), striving to portray scenes of modern life, landscapes, and fleeting moments with loose brushwork and vibrant palettes. Their approach broke away from the somber tones and historical subjects favored by the Salon, the official art exhibition of the Académie des Beaux-Arts.
            
            Despite early criticism and rejection by traditional art institutions, Impressionism profoundly influenced subsequent movements like Post-Impressionism and Fauvism. Moreover, it democratized art by focusing on everyday subjects and ordinary people rather than aristocratic or mythological themes.
            
            Beyond painting, Impressionism’s impact extended into literature and music, inspiring writers such as Marcel Proust and composers like Claude Debussy, who echoed its emphasis on atmosphere and mood.
            
            Passage:
            The Rise and Global Impact of Streaming Services on the Film Industry
            
            In the early 21st century, streaming platforms such as Netflix, Amazon Prime Video, and later Disney+, fundamentally transformed the film and television industry. Initially starting as a DVD-by-mail service, Netflix pivoted to streaming in 2007, leading a digital revolution that disrupted the traditional distribution models of cinema and broadcast television.
            
            One of the most significant shifts caused by streaming has been the collapse of the theatrical release window. Previously, films would debut exclusively in cinemas for months before becoming available for home viewing. Streaming services, however, began offering original films directly to subscribers, sometimes releasing them simultaneously in theaters and online—most notably during the COVID-19 pandemic, which accelerated this trend.
            
            This shift has sparked debates across the industry. While some filmmakers argue that streaming democratizes access and offers greater creative freedom, others worry it diminishes the cultural experience of cinema and undermines box office revenue. Additionally, streaming platforms use complex algorithms to personalize recommendations, influencing viewer habits and even greenlighting content based on data-driven predictions rather than purely artistic vision.
            
            Globally, the reach of streaming has enabled regional and independent films to gain international exposure. Korean dramas, Spanish thrillers, and Indian series now enjoy mainstream success beyond their borders, contributing to a diversification of storytelling and audience tastes worldwide.
            
            Passage:
            Largest Automakers in the World 2025 by Production Volume
            According to industry reports, the top three largest automobile manufacturers by production volume in 2025 are:
            Toyota Motor Corporation - Producing approximately 10.5 million vehicles annually, Toyota remains the largest automaker globally. Headquartered in Japan, Toyota is known for its focus on hybrid and fuel-efficient vehicles.
            Volkswagen Group - Producing about 9.8 million vehicles, Volkswagen maintains a strong global presence with multiple brands including Audi, Porsche, and Skoda.
            General Motors - With a production volume of around 6.5 million vehicles, GM is the largest American automaker, emphasizing electric vehicle development in recent years.
            
            Question: In what ways have streaming platforms altered both the distribution of films and the global cultural landscape of entertainment?
            Answer:""",
            
            """Fill in the missing code given below.
            
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import classification_report, confusion_matrix
            
            # Load dataset
            df = pd.read_csv('heart.csv')
            
            # Check for missing values
            if df.isnull().sum().sum() > 0:
                df = df.dropna()
            
            # Define features and target
            X = df.drop('target', axis=1)
            y = df['target']
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Standardize the feature columns
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Initialize the model
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Evaluate the model
            
            Next line of code:""",
        ]
    },
}


@register_evaluator(
    "text"
)
class TextEvaluator(BaseEvaluator):
    def __init__(
        self,
        base_model: Any = None,
        tokenizer: Any = None,
        gt_data: str = None,
        test_data: Union[str, list] = None,
        metrics="similarity",
        similarity_model_id: str = "sentence-transformers/all-mpnet-base-v2",
        max_new_tokens=128,
        crop_question=True,
        num_samples=None,
        language="en",
        gen_answer_fn=None,
        generation_config=None,
        generation_config_base=None,
        seqs_per_request=None,
        use_chat_template=None,
    ) -> None:
        assert (
            base_model is not None or gt_data is not None
        ), "Text generation pipeline for evaluation or ground trush data must be defined"

        self.test_data = test_data
        self.metrics = metrics
        self.max_new_tokens = max_new_tokens
        self.tokenizer = tokenizer
        self._crop_question = crop_question
        self.num_samples = num_samples
        self.generation_config = generation_config
        self.generation_config_base = generation_config
        self.seqs_per_request = seqs_per_request
        self.generation_fn = gen_answer_fn
        self.use_chat_template = use_chat_template
        if self.generation_config is not None:
            assert self.seqs_per_request is not None

        # Take language from the base model if provided
        self.language = language

        if base_model:
            self.gt_data = self._generate_data(
                base_model, gen_answer_fn, generation_config=generation_config
            )
        else:
            self.gt_data = pd.read_csv(gt_data, keep_default_na=False)

        # Take language ground truth if no base model provided
        if self.language is None and "language" in self.gt_data.columns:
            self.language = self.gt_data["language"].values[0]

        self.similarity = None
        self.divergency = None
        if "similarity" in self.metrics:
            self.similarity = TextSimilarity(similarity_model_id)
        if "divergency" in self.metrics:
            assert tokenizer is not None
            self.divergency = TextDivergency(tokenizer)

        self.last_cmp = None

    def get_generation_fn(self):
        return self.generation_fn

    def score(self, model_or_data, gen_answer_fn=None, **kwargs):
        if isinstance(model_or_data, str) and os.path.exists(model_or_data):
            predictions = pd.read_csv(model_or_data, keep_default_na=False)
        else:
            predictions = self._generate_data(model_or_data, gen_answer_fn, self.generation_config)
        self.predictions = predictions

        all_metrics_per_prompt = {}
        all_metrics = {}

        if self.similarity:
            metric_dict, metric_per_question = self.similarity.evaluate(
                self.gt_data, predictions
            )
            all_metrics.update(metric_dict)
            all_metrics_per_prompt.update(metric_per_question)

        if self.divergency:
            metric_dict, metric_per_question = self.divergency.evaluate(
                self.gt_data, predictions
            )
            all_metrics.update(metric_dict)
            all_metrics_per_prompt.update(metric_per_question)

        self.last_cmp = all_metrics_per_prompt
        self.last_cmp["prompts"] = predictions["prompts"].values
        self.last_cmp["source_model"] = self.gt_data["answers"].values
        self.last_cmp["optimized_model"] = predictions["answers"].values
        self.last_cmp = pd.DataFrame(self.last_cmp)
        self.last_cmp.rename(columns={"prompts": "prompt"}, inplace=True)

        return pd.DataFrame(all_metrics_per_prompt), pd.DataFrame([all_metrics])

    def worst_examples(self, top_k: int = 5, metric="similarity"):
        assert self.last_cmp is not None

        if metric in ["SDT", "SDT norm"]:
            res = self.last_cmp.nlargest(top_k, metric)
        else:
            res = self.last_cmp.nsmallest(top_k, metric)

        res = list(row for idx, row in res.iterrows())

        return res

    def _generate_data(self, model, gen_answer_fn=None, generation_config=None):
        def default_gen_answer(model, tokenizer, prompt, max_new_tokens, crop_question, use_chat_template=False):
            is_awq = getattr(model, "is_awq", None) is not None
            device = "cpu"
            if hasattr(model, "device"):
                device = model.device

            if use_chat_template:
                message = [{"role": "user", "content": prompt}]
                inputs = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

            if is_awq:
                with patch_awq_for_inference(is_awq):
                    tokens = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, **get_ignore_parameters_flag())
            else:
                tokens = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, **get_ignore_parameters_flag())
            if crop_question:
                tokens = tokens[:, inputs["input_ids"].shape[-1] :]
            return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

        gen_answer_fn = gen_answer_fn or default_gen_answer

        if self.test_data:
            if isinstance(self.test_data, str):
                data = pd.read_csv(self.test_data)
            else:
                if isinstance(self.test_data, dict):
                    assert "prompts" in self.test_data
                    data = dict(self.test_data)
                else:
                    data = {"prompts": list(self.test_data)}
                data = pd.DataFrame.from_dict(data)
        else:
            data = pd.DataFrame.from_dict(default_data[self.language])

        prompt_data = data["prompts"]

        answers = []
        prompts = (
            prompt_data.values
            if self.num_samples is None
            else prompt_data.values[: self.num_samples]
        )

        if generation_config is None:
            for p in tqdm(prompts, desc="Evaluate pipeline"):
                answers.append(
                    gen_answer_fn(
                        model,
                        self.tokenizer,
                        p,
                        self.max_new_tokens,
                        self._crop_question,
                        self.use_chat_template
                    )
                )
        else:
            with tqdm(total=len(prompt_data.values)) as progress_bar:
                batch = []
                for p_idx, p in enumerate(prompt_data.values):
                    progress_bar.update(1)
                    batch.append(p)
                    if (
                        len(batch) == self.seqs_per_request
                        or p_idx == len(prompt_data.values) - 1
                    ):
                        ans_batch = model.generate(
                            batch, [generation_config] * len(batch)
                        )
                        for ans in ans_batch:
                            answers.append(ans.m_generation_ids[0])

                        batch.clear()

        res_data = {"prompts": list(prompts), "answers": answers}
        df = pd.DataFrame(res_data)
        df["language"] = self.language

        return df
