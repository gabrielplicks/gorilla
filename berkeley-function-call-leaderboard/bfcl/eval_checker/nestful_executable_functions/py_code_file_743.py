import re



def get_img_srcs_from_html(html_string: str) -> list[str]:

    """Extracts all image `src` attributes from `<img>` tags in an HTML string.



    Args:

        html_string: The HTML string to extract image `src` attributes from.



    Returns:

        A list of strings containing the `src` attributes.

    """

    img_tags = re.findall(r'<img.*?src="(.*?)".*?>', html_string, re.DOTALL)

    img_srcs = [img_tag.split('"')[0] for img_tag in img_tags]

    return img_srcs

