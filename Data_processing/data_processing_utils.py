from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import emoji
import re
import string


def classify_products(products):
    """
        Classify products based on product categories.
        Products -> Main Category -> Sub Category -> Product Category
        Args:
            products: products dataframe

        Returns:
            productID_classID: dict, product ID -> class ID
            className_classID: dict, class name -> class ID
            calssName_productID: dict, class name -> product ID
            classID_className: dict, class ID -> calss name
    """
    productID_classID = {}
    className_classID = {}  # index each class
    className_productID = defaultdict(list)  # store the productIDs in corresponding classes
    curr_num_classes = 0

    # map productID to class ID
    for idx, row_data in products.iterrows():
        class_name = ':'.join([row_data.main_category, row_data.sub_category,
                               row_data.product_category])
        if class_name not in className_classID:
            className_classID[class_name] = curr_num_classes
            curr_num_classes += 1

        product_ID, classID = row_data.loc['product_id'], className_classID[class_name]
        productID_classID[product_ID] = classID
        className_productID[class_name].append(product_ID)

    # reverse mapping, class ID -> class name
    classID_className = {value: key for key, value in className_classID.items()}
    return productID_classID, className_classID, className_productID, classID_className


def plot_distribution(media, classID_className, multiLabels=True):
    """
        Args:
            media: dataframe, tagged media
            classID_className: dict, class ID -> class Name, for bar plot

        Returns:
            distribution:
    """
    # count the number of mentions of each class
    classID_numMentions = defaultdict(int)
    for idx, row_data in media.iterrows():
        if multiLabels:
            for classID in row_data.label:
                classID_numMentions[classID] += 1
        else:
            classID = row_data.class_ID
            classID_numMentions[classID] += 1

    classID_mentions = [[key, val] for key, val in classID_numMentions.items()]
    classID_mentions.sort(key=lambda x: x[1], reverse=True)

    distribution = [i for i in classID_mentions if i[1] > 1131]
    #distribution = [i for i in classID_mentions]

    #med_num = np.median(np.array([i[1] for i in distribution]))
    #print("median of number of products in each product category is: %d" %med_num)
    class_names = [classID_className[i[0]].split(':')[2] for i in distribution]
    sns.set(font_scale=1.8)
    plt.figure(figsize=(15, 8))
    ax = sns.barplot(class_names, [i[1] for i in distribution])
    for item in ax.get_xticklabels():
        item.set_rotation(90)
    plt.title("distribution of media posts", fontsize=24)
    plt.ylabel('number of media posts', fontsize=26)
    plt.xlabel('product type', fontsize=26)

    # add the text labels for max and min numbers
    rects = [ax.patches[0], ax.patches[-1]]
    labels = [distribution[0][1], distribution[-1][1]]
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 10, label, ha='center', va='bottom', fontsize=22)
    # plt.show()
    return distribution


def keepGloVeWords(s, GloVe_words):
    """ Further process captions, only keep first MAX_WORDS_IN_CAPTION words found in GloVe.
        Probaly not too useful to embed those random words in captions, since most of them just appear once
        and contain no semantics.
    """
    MAX_WORDS_IN_CAPTION = 40
    individual_words = s.split()
    res = []
    for word in individual_words:
        if word in GloVe_words:
            res.append(word)
        if len(res) == MAX_WORDS_IN_CAPTION:
            break
    return " ".join(res)


def isInGlove(s):
    """ Test if a string consists only of lower-case English words"""
    for char in s:
        if char not in string.ascii_lowercase:
            return False
    return True


def cleaner(text):
    # remove punctuation
    remove_punct = str.maketrans('', '', string.punctuation.replace('#', ''))
    text = text.translate(remove_punct).lower()
    return text


def remove_emoji(string):
    emojies = list(emoji.UNICODE_EMOJI)
    emojies = '|'.join((emojies))
    emojies = emojies.replace("|*", "")
    otherchar = "|«" + "|»" + "|•" + "|•••" + "|••••" + "|♡" + "|⚬" + "|✓" + "|・" + "|ㅡ" + "|🇵🇭" + "|🇦🇺" + "|🇦🇺" + "|🇧🇷" + "|🇨🇦" + "|🇩🇪" + "|🇪🇸" + "|🇫🇷" + "|🇬🇧" + "|🇮🇹" + "|🇲🇽" + "|🇵🇭" + "|🇺🇸" + "|🇷🇺" + '|™' + '|®' + "|🏻🏼🏽🏾🏿" + "|🏻‍" + "|🏻" + "|🏼‍️" + "|🏼️" + "|🏽" + "|🏾" + '|’' + '|‘' + '|“' + '|”' + '|–' + '|—' + '|—' + "|'" + '|"'
    emojies += otherchar
    return re.sub('  ', ' ', re.sub(emojies, '', string))


def cleanunderscore(s):
    s = s.replace('_', '')
    return s


def normalize(s):
    """
    Given a text, cleans and normalizes it.
    """
    s = s.lower()
    # Replace ips
    s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)
    # Isolate punctuation
    s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Replace numbers and symbols with language
    s = s.replace('&', ' and ')
    s = s.replace('#', ' mention ')
    s = s.replace('@', ' at ')
    s = s.replace('0', ' zero ')
    s = s.replace('1', ' one ')
    s = s.replace('2', ' two ')
    s = s.replace('3', ' three ')
    s = s.replace('4', ' four ')
    s = s.replace('5', ' five ')
    s = s.replace('6', ' six ')
    s = s.replace('7', ' seven ')
    s = s.replace('8', ' eight ')
    s = s.replace('9', ' nine ')
    return s


def under_over_sample(orig_media, under_sample_ratio=0.2, over_sample_ratio=3.2):
    #media = pd.DataFrame()

    media = orig_media[orig_media.under_sample == 0]
    media = media[media.over_sample == 0]

    both = orig_media[orig_media.under_sample == 1]
    both = both[both.over_sample == 1]

    media = media.append(both)

    underSample_instances = orig_media[orig_media.under_sample == 1]
    underSample_instances = underSample_instances[underSample_instances.over_sample == 0]
    # overSample == 1 and underSample == 0
    overSample_instances = orig_media[orig_media.over_sample == 1]
    overSample_instances = overSample_instances[overSample_instances.under_sample == 0]

    underSampled = underSample_instances.sample(frac=under_sample_ratio, axis=0, random_state=42)
    overSampled = overSample_instances.sample(frac=over_sample_ratio, axis=0, random_state=42, replace=True)
    #overSampled = overSample_instances
    media = media.append(underSampled)
    media = media.append(overSampled)
    return media
