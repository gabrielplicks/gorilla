from typing import Tuple



def build_facebook_url(username: str, post_id: int) -> str:

    """Builds a Facebook URL string given a username and post_id.



    Args:

        username: The username of the Facebook user.

        post_id: The ID of the post.



    Returns:

        A Facebook URL string in the format:

        https://www.facebook.com/username/posts/post_id

    """

    return f"https://www.facebook.com/{username}/posts/{post_id}"

