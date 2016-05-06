import sys, urllib2, re, string, time, threading, wikipedia, warnings
warnings.filterwarnings("ignore")
def get_random_wikipedia_article():
    condition = True
    while condition:
        try:
            articletitle = wikipedia.random(1)
            content = wikipedia.page(title=articletitle).content
            condition = False
        except:
            condition = True
            
    return(content, articletitle)

class WikiThread(threading.Thread):
    articles = list()
    articlenames = list()
    lock = threading.Lock()

    def run(self):
        (article, articlename) = get_random_wikipedia_article()
        WikiThread.lock.acquire()
        WikiThread.articles.append(article)
        WikiThread.articlenames.append(articlename)
        WikiThread.lock.release()

def get_random_wikipedia_articles(n):
    """
    Downloads n articles in parallel from Wikipedia and returns lists
    of their names and contents. Much faster than calling
    get_random_wikipedia_article() serially.
    """
    maxthreads = 8
    WikiThread.articles = list()
    WikiThread.articlenames = list()
    wtlist = list()
    for i in range(0, n, maxthreads):
        print 'downloaded %d/%d articles...' % (i, n)
        for j in range(i, min(i+maxthreads, n)):
            wtlist.append(WikiThread())
            wtlist[len(wtlist)-1].start()
        for j in range(i, min(i+maxthreads, n)):
            wtlist[j].join()
    return (WikiThread.articles, WikiThread.articlenames)



if __name__ == '__main__':
    t0 = time.time()

    (articles, articlenames) = get_random_wikipedia_articles(1)
    for i in range(0, len(articles)):
        print "-"*50
        print articlenames[i]
        print "-"*50
        print articles[i][:500]
    t1 = time.time()
    print 'took %f' % (t1 - t0)