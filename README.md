## Astrophotography tools

Work in progress.
If just reading raw images, you should just use
[rawkit](https://github.com/photoshell/rawkit/), for example:
```python
    from rawkit.raw import Raw
    im = Raw(filename)
    raw = im.raw_image()
    print(raw.shape)
    print(raw[0])
```

For this library, something like this should work:

    In [1]: from astrotools.readers.readers import AstroImage

    In [2]: filename = "/home/julienl/astropictures/170603-nick/IMG_9655.CR2"

    In [3]: im = AstroImage(filename)

    In [4]: cim  = im.image

    In [5]: cim.shape
    Out[5]: (3, 3476, 5208)

